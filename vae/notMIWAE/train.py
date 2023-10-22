# train.py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from utils import MyModel, compute_loss, train_step, softmax

# Add argument parser to handle command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fdata', help='Full data CSV file path', required=True)
parser.add_argument('--mdata', help='Observed data CSV file path', required=True)
args = parser.parse_args()

N, D = data.shape
n_latent = D - 1
n_hidden = 128
n_samples = 20
max_iter = 30000
batch_size = 16

batch_pointer = 0
start = time.time()
best = float("inf")

training_losses = []
validation_losses = []

model = MyModel(D, n_hidden, n_latent)
optimizer = tf.keras.optimizers.Adam()

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

model.save('./trained_model.h5')
