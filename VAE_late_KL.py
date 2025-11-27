import tensorflow as tf
import numpy as np
import sys
import os

# =========================================================
# Config
# =========================================================
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

training_data = sys.argv[1]  # Path to TFRecord file
test_data_path = sys.argv[2]  # Path to TFRecord file
save_name=sys.argv[3]
mode=sys.argv[4]
beta_input = float(sys.argv[5])
input_shape = (32, 32, 32, 1)
z_dim = 8
learning_rate = 3e-4
epochs = 30
batchsize = 64

# =========================================================
# TFRecord Parser
# =========================================================
def parse_tfrecord(example_proto):
    features = {'data': tf.io.FixedLenFeature([32768], tf.float32)}
    parsed = tf.io.parse_single_example(example_proto, features)
    data = tf.reshape(parsed['data'], input_shape)
    return data, data

# =========================================================
# Datasets
# =========================================================
def load_dataset(path, batchsize, seed):
    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=3200, seed=seed)
    ds = ds.batch(batchsize)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()
    return ds

train_ds = load_dataset(training_data, batchsize, seed)
test_ds = load_dataset(test_data_path, batchsize, seed)

# =========================================================
# VAE model
# =========================================================
class VAE(tf.keras.Model):
    def __init__(self, latent_dim, encoder, decoder):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(0.5 * logvar) + mean

    def decode(self, z):
        return self.decoder(z)

# =========================================================
# Encoder & Decoder
# =========================================================
def build_encoder():
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Conv3D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.MaxPool3D(2)(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv3D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Flatten()(x)
    z = tf.keras.layers.Dense(z_dim * 2, activation=None)(x)
    return tf.keras.Model(inputs, z, name='encoder')

def build_decoder():
    latent_inputs = tf.keras.Input(shape=(z_dim,))
    x = tf.keras.layers.Dense(8 * 8 * 8 * 32, activation='relu', kernel_initializer='he_normal')(latent_inputs)
    x = tf.keras.layers.Reshape((8, 8, 8, 32))(x)
    x = tf.keras.layers.Conv3DTranspose(64, 5, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Conv3DTranspose(128, 5, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    outputs = tf.keras.layers.Conv3D(1, 3, padding='same', activation='tanh')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')

# =========================================================
# Loss Functions
# =========================================================
def compute_KL_prior_latent(latent_mean, latent_std, epsilon_loss=1e-8):
    latent_std = tf.maximum(latent_std, epsilon_loss)
    kl_div = 0.5 * tf.reduce_sum(
        tf.square(latent_mean) + tf.square(latent_std) - tf.math.log(tf.square(latent_std)) - 1,
        axis=1,
    )
    return tf.reduce_mean(kl_div)

def compute_loss(model, x, beta=0):
    mean, logvar = model.encode(x)
    latent_std = tf.exp(0.5 * logvar)
    z = model.reparameterize(mean, logvar)
    x_pred = model.decode(z)
    reconstruction_loss = tf.reduce_mean(tf.square(x - x_pred), axis=[1, 2, 3, 4])
    kl_loss = compute_KL_prior_latent(mean, latent_std)* beta
    total_loss = reconstruction_loss + beta * kl_loss
    return tf.reduce_mean(total_loss), tf.reduce_mean(reconstruction_loss), tf.reduce_mean(kl_loss)

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss, mse, KL = compute_loss(model, x,0)
    gradients = tape.gradient(loss, model.trainable_variables)
    # inside train step get global grad norm
    grad_norm = tf.linalg.global_norm(gradients)
#    tf.print("grad_norm:", grad_norm)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, mse, KL


@tf.function
def train_step_KL(model, x, optimizer,beta):
    with tf.GradientTape() as tape:
        loss, mse, KL = compute_loss(model, x, beta)
    gradients = tape.gradient(loss, model.trainable_variables)
    # inside train step get global grad norm
    grad_norm = tf.linalg.global_norm(gradients)
    #tf.print("grad_norm:", grad_norm)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, mse, KL
# =========================================================
# Build and Train
# =========================================================
encoder = build_encoder()
decoder = build_decoder()
vae = VAE(z_dim, encoder, decoder)

optimizer = tf.keras.optimizers.Adam(learning_rate)
num_train_samples = sum(1 for _ in tf.data.TFRecordDataset(training_data))
steps_per_epoch = num_train_samples // batchsize

os.makedirs(f'{save_name}_log', exist_ok=True)

import os
import numpy as np

log_path = f"{save_name}_log/log.txt"
os.makedirs(f"{save_name}_log", exist_ok=True)

with open(log_path, 'a') as log_file:
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_losses, epoch_mses, epoch_kls = [], [], []

        # Determine beta based on mode and epoch
        if mode == 'constant':
            beta = beta_input  # Active all epochs
        elif mode == 'late':
            beta = beta_input if epoch >= 10 else 0.0  # Activate after epoch 10
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Training loop
        for step, (x_batch, _) in enumerate(train_ds.take(steps_per_epoch)):
            loss, mse, KL = train_step_KL(vae, x_batch, optimizer, beta)
            epoch_losses.append(loss)
            epoch_mses.append(mse)
            epoch_kls.append(KL)

            if step % 10 == 0:
                log_file.write(f"Epoch {epoch+1}, Step {step}, "
                               f"Loss: {loss:.5f}, MSE: {mse:.5f}, KL: {KL:.5f}\n")
                log_file.flush()
                os.fsync(log_file.fileno())
                print(f"Step {step}/{steps_per_epoch}, Loss: {loss:.5f}, "
                      f"MSE: {mse:.5f}, KL: {KL:.5f}")

        # Evaluate on test set
        test_loss, test_mse, test_kl = 0.0, 0.0, 0.0
        test_steps = 11
        for x_test, _ in test_ds.take(test_steps):
            l, m, k = compute_loss(vae, x_test, beta)
            test_loss += l
            test_mse += m
            test_kl += k
        test_loss /= test_steps
        test_mse /= test_steps
        test_kl /= test_steps

        # Log epoch summary
        log_file.write(f"Epoch {epoch+1} Summary -> "
                       f"TrainLoss: {np.mean(epoch_losses):.5f}, "
                       f"TestLoss: {test_loss:.5f}, TestMSE: {test_mse:.5f}, "
                       f"TestKL: {test_kl:.5f}\n")
        log_file.flush()
        os.fsync(log_file.fileno())

        print(f"Epoch {epoch+1} finished. TrainLoss: {np.mean(epoch_losses):.5f}, "
              f"TestLoss: {test_loss:.5f}")

        # Save model checkpoint
        vae.save_weights(f"{save_name}_log/vae_epoch_{epoch+1}.weights.h5")

# =========================================================
# Final Model Save
# =========================================================
vae.encoder.save(f'{save_name}_log/encoder_model.keras')
vae.decoder.save(f'{save_name}_log/decoder_model.keras')
vae.save_weights(f'{save_name}_log/vae_final.weights.h5')
print(f"Models saved in {save_name}_log")

