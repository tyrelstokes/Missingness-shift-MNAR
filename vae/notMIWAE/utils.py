import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers

# not MIWAE model framework
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


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
