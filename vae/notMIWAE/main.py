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
from utils import MyModel, compute_loss, train_step, softmax


def imputationRMSE(model, Xorg, Xnan, L):
    N = len(Xorg)

    Xz = Xnan.copy()
    Xz[np.isnan(Xnan)] = 0
    S = np.array(~np.isnan(Xnan), dtype=np.float32)

    def imp(xz, s, L):
        p_x_given_z, q_z, l_z = model(xz[None, :], s[None, :], L)
        log_p_x_given_z = tf.reduce_sum(s[None, :] * p_x_given_z.log_prob(xz[None, :]), axis=-1)
        log_p_z = tf.reduce_sum(tfp.distributions.Normal(loc=0.0, scale=1.0).log_prob(l_z), axis=-1)
        log_q_z_given_x = tf.reduce_sum(q_z.log_prob(l_z), axis=-1)
        wl = softmax((log_p_x_given_z + log_p_z - log_q_z_given_x).numpy())
        _mu = p_x_given_z.mean().numpy()
        xm = np.sum((_mu * wl.T).T, axis=1)
        xmix = xz + xm * (1 - s)
        return _mu, wl, xm, xmix

    XM = np.zeros_like(Xorg)
    # Here we combine the imputed values with the original known values
    Ximp = np.where(np.isnan(Xnan), XM, Xnan)
    # Compute RMSE over imputed values only
    imputed_values = XM[np.isnan(Xnan)]
    original_values = Xorg[np.isnan(Xnan)]
    rmse = np.sqrt(((imputed_values - original_values) ** 2).mean())
    return rmse, Ximp

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

model = tf.keras.models.load_model('trained_model.h5')

data = np.array(pd.read_csv(args.observed_data, usecols = ['xobs_1', 'xobs_2', 'xobs_3']))     # (1000, 3)
data_full = np.array(pd.read_csv(args.full_data, usecols = ['x_1', 'x_2', 'x_3']))         # (1000, 3)

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
output_dir = "./output/"
try:
    os.makedirs(output_dir, exist_ok=True)
except OSError as error:
    print(f"Creation of the directory {output_dir} failed: {error}")
else:
    print(f"Successfully created the directory {output_dir}")

df_imputations = pd.DataFrame(imputations)
df_imputations.to_csv(os.path.join(output_dir, 'data_imputation.csv'), index=False)
