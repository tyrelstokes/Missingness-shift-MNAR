import numpy as np
import os
import tensorflow as tf
import tensorflow_probability as tfp
import keras
import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse
from tensorflow.keras import layers
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Nvidia GPU resources
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


"""**notMIWAE/MIWAE in our observed dataset**"""

# Add argument parser to handle command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fdata', help='Full data CSV file path', required=True)
parser.add_argument('--mdata', help='Observed data CSV file path', required=True)
args = parser.parse_args()

data = np.array(pd.read_csv(args.mdata, usecols = ['xobs_1', 'xobs_2', 'xobs_3']))     # (1000, 3)
data_full = np.array(pd.read_csv(args.fdata, usecols = ['x_1', 'x_2', 'x_3']))         # (1000, 3)

N, D = data.shape
n_latent = D - 1
n_hidden = 128
n_samples = 20
max_iter = 30000
batch_size = 16

# ---- standardize data
# data = data - np.mean(data, axis=0)
# data = data / np.std(data, axis=0)
# order = np.arange(N)
# ---- random permutation
p = np.random.permutation(N)
data = data[p, :]
data_full = data_full[p, :]

# ---- we use the full dataset for training here, but you can make a train-val split
Xtrain = data_full.copy()
Xval = Xtrain.copy()

# ---- introduce missing process
Xnan = data.copy()
Xz = Xnan.copy()
Xz[np.isnan(Xnan)] = 0
S = np.array(~np.isnan(Xnan), dtype=np.float32)

class MyModel(tf.keras.Model):
    def __init__(self, D, n_hidden, n_latent):
        super(MyModel, self).__init__()
        self.D = D
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.logstd = tf.Variable(initial_value=0.0, trainable=True, dtype=tf.float32)
        self.dense1 = layers.Dense(units=self.n_hidden, activation=tf.nn.tanh, name='l_enc1')
        self.dense2 = layers.Dense(units=self.n_hidden, activation=tf.nn.tanh, name='l_enc2')
        self.q_mu = layers.Dense(units=self.n_latent, activation=None, name='q_mu')
        self.q_logstd = layers.Dense(units=self.n_latent, activation=lambda x: tf.clip_by_value(x, -10, 10), name='q_logstd')
        self.mu = layers.Dense(units=self.D, activation=None, name='mu')
        self.W = tf.Variable(initial_value=tf.zeros([1, 1, self.D]), trainable=True, dtype=tf.float32)
        self.b = tf.Variable(initial_value=tf.zeros([1, 1, self.D]), trainable=True, dtype=tf.float32)

    def call(self, x, s, n_samples):
        s_pl = tf.expand_dims(1 - s, axis=1)
        x_pl = tf.expand_dims(x * s, axis=1)
        x = self.dense1(x)
        x = self.dense2(x)
        q_mu = self.q_mu(x)
        q_logstd = self.q_logstd(x)
        q_z = tfp.distributions.Normal(loc=q_mu, scale=tf.exp(q_logstd))
        l_z = q_z.sample(n_samples)
        l_z = tf.transpose(l_z, perm=[1, 0, 2])
        mu = self.mu(l_z)
        p_x_given_z = tfp.distributions.Normal(loc=mu, scale=tf.exp(self.logstd))
        l_out_mixed = mu * s_pl + x_pl

        W = -tf.nn.softplus(self.W)
        b = self.b
        logits = W * (l_out_mixed - b)
        p_s_given_x = tfp.distributions.Bernoulli(logits=logits)
        return p_x_given_z, p_s_given_x, q_z, l_z

model = MyModel(D, n_hidden, n_latent)
optimizer = tf.keras.optimizers.Adam()

# loss computation
@tf.function
def compute_loss(x, s, n_samples, model):
    p_x_given_z, p_s_given_x, q_z, l_z = model(x, s, n_samples)
    prior = tfp.distributions.Normal(loc=0.0, scale=1.0)
    # evaluate the observed data in p(x|z)
    log_p_x_given_z = tf.reduce_sum(tf.expand_dims(s, axis=1) * p_x_given_z.log_prob(tf.expand_dims(x, axis=1)), axis=-1)
    # evaluate the z-samples in q(z|x)
    q_z2 = tfp.distributions.Normal(loc=tf.expand_dims(q_z.loc, axis=1), scale=tf.expand_dims(q_z.scale, axis=1))
    log_q_z_given_x = tf.reduce_sum(q_z2.log_prob(l_z), axis=-1)
    # evaluate the z-samples in the prior p(z)
    log_p_z = tf.reduce_sum(prior.log_prob(l_z), axis=-1)
    # evaluate the mask in p(s|x)
    log_p_s_given_x = tf.reduce_sum(p_s_given_x.log_prob(tf.expand_dims(s, axis=1)), axis=-1)

    lpxz = log_p_x_given_z
    lpz = log_p_z
    lqzx = log_q_z_given_x
    lpsx = log_p_s_given_x

    # MIWAE
    # importance weights
    l_w = lpxz + lpz - lqzx
    # sum over samples
    log_sum_w = tf.reduce_logsumexp(l_w, axis=1)
    # average over samples
    log_avg_weight = log_sum_w - tf.math.log(tf.cast(n_samples, tf.float32))

    # average over minibatch to get the average llh
    MIWAE = tf.reduce_mean(log_avg_weight, axis=-1)

    # not-MIWAE
    # importance weights
    l_w = lpxz + lpsx + lpz - lqzx
    # sum over samples
    log_sum_w = tf.reduce_logsumexp(l_w, axis=1)
    # average over samples
    log_avg_weight = log_sum_w - tf.math.log(tf.cast(n_samples, tf.float32))
    # average over minibatch to get the average llh
    notMIWAE = tf.reduce_mean(log_avg_weight, axis=-1)

    # use MIWAE or notMIWAE as loss.
    return -notMIWAE

# This function performs one step of optimization
@tf.function
def train_step(x, s, n_samples, model, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(x, s, n_samples, model)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

batch_pointer = 0
start = time.time()
best = float("inf")

training_losses = []
validation_losses = []

# Training loop
for i in range(max_iter):
    x_batch = Xz[batch_pointer: batch_pointer + batch_size, :]
    s_batch = S[batch_pointer: batch_pointer + batch_size, :]
    x_batch = tf.cast(x_batch, tf.float32)
    loss = train_step(x_batch, s_batch, n_samples, model, optimizer)
    training_losses.append(loss)

    batch_pointer += batch_size

    if batch_pointer > N - batch_size:
        batch_pointer = 0
        p = np.random.permutation(N)
        Xz = Xz[p, :]
        S = S[p, :]

    if i % 100 == 0:
        took = time.time() - start
        start = time.time()
        x_batch = Xz
        s_batch = S
        x_batch = tf.cast(x_batch, tf.float32)
        val_loss = compute_loss(x_batch, s_batch, n_samples, model)
        validation_losses.append(val_loss)
        print(f"{i}/{max_iter} updates, {took:.2f} s, {loss:.2f} train_loss, {val_loss:.2f} val_loss")

"""**Get the imputation values and RMSE**
This approach assumes that the data is missing completely at random (MCAR), which means that the probability of a value being missing does not depend on the observed or missing data. If missing data has a different pattern (e.g., missing at random, MAR, or not missing at random, NMAR), we might need a different approach.
"""

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def not_imputationRMSE(model, Xorg, Xnan, L):
    N = len(Xorg)

    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0
    S = np.array(~np.isnan(Xnan), dtype=np.float32)

    def imp(xz, s, L):
        p_x_given_z, p_s_given_x, q_z, l_z = model(xz[None, :], s[None, :], L)
        log_p_x_given_z = tf.reduce_sum(s[None, :] * p_x_given_z.log_prob(xz[None, :]), axis=-1)
        log_p_s_given_x = tf.reduce_sum(p_s_given_x.log_prob(s[None, :]), axis=-1)
        log_p_z = tf.reduce_sum(tfp.distributions.Normal(loc=0.0, scale=1.0).log_prob(l_z), axis=-1)
        log_q_z_given_x = tf.reduce_sum(q_z.log_prob(l_z), axis=-1)
        wl = softmax((log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x).numpy())
        _mu = p_x_given_z.mean().numpy()
        xm = np.sum((_mu * wl.T).T, axis=1)
        xmix = xz + xm * (1 - s)
        return _mu, wl, xm, xmix

    XM = np.zeros_like(Xorg)
    for i in range(N):
        xz = Xz[i, :]
        s = S[i, :]
        _mu, wl, xm, xmix = imp(xz, s, L)
        XM[i, :] = np.squeeze(xm)

    # Here we combine the imputed values with the original known values
    Ximp = np.where(np.isnan(Xnan), XM, Xnan)

    # Compute RMSE over imputed values only
    imputed_values = XM[np.isnan(Xnan)]
    original_values = Xorg[np.isnan(Xnan)]
    rmse = np.sqrt(((imputed_values - original_values) ** 2).mean())
    return rmse, Ximp


data = np.array(pd.read_csv(args.mdata, usecols = ['xobs_1', 'xobs_2', 'xobs_3']))     # (1000, 3)
data_full = np.array(pd.read_csv(args.fdata, usecols = ['x_1', 'x_2', 'x_3']))         # (1000, 3)

N, D = data.shape
n_latent = D - 1
n_hidden = 128
n_samples = 20
max_iter = 30000
batch_size = 16

# ---- standardize data
# data = data - np.mean(data, axis=0)
# data = data / np.std(data, axis=0)
order = np.arange(N)
# ---- random permutation
p = np.random.permutation(N)
order = order[p]
data = data[p, :]
data_full = data_full[p, :]

# ---- we use the full dataset for training here, but you can make a train-val split
Xtrain = data_full.copy()
Xval = Xtrain.copy()

# ---- introduce missing process
Xnan = data.copy()
Xz = Xnan.copy()
Xz[np.isnan(Xnan)] = 0
S = np.array(~np.isnan(Xnan), dtype=np.float32)

rmse, imputations = not_imputationRMSE(model, Xtrain, Xnan, 10000)
print("imputation RMSE: ", rmse)

restore_order = np.argsort(order)
imputations = imputations[restore_order, :]

"""**Output imputations saved as csv file under the same path**"""
output_dir = "vae/python-data/"
try:
    os.makedirs(output_dir, exist_ok=True)
except OSError as error:
    print(f"Creation of the directory {output_dir} failed: {error}")
else:
    print(f"Successfully created the directory {output_dir}")

df_imputations = pd.DataFrame(imputations)
df_imputations.to_csv(os.path.join(output_dir, 'data_imputation.csv'), index=False)
