import tensorflow as tf
import numpy as tf
from model import *
from config import *

latent_dim = model_config.latent_dim
image_shape = model_config.image_shape
gamma = train_config.gp_gamma

gan = GAN()
# Accelerated WGAN - GP
def train_step(x, prev_dis_loss, prev_gen_loss):
    batch_size = x.shape[0]
    r_d, r_g = 0, 0

    with tf.GradientTape() as dis_tape:
        latent = tf.random.normal(shape=[batch_size, latent_dim])
        fake = gan.generator(latent)

        fake_output = gan.discriminator(fake)
        real_output = gan.discriminator(x)

        dis_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

        epsilon = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
        x_hat = epsilon * fake + (1. - epsilon) * x
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = gan.discriminator(x_hat)
        gradients = t.gradient(d_hat, [x_hat])[0]
        grad_norm = tf.sqrt(1e-10 + tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gp = tf.reduce_mean((grad_norm - 1.) ** 2)  # mean, sum을 사용했다.
        dis_loss += gp * 10.

    r_d = tf.abs(dis_loss - prev_dis_loss) / (prev_dis_loss + 1e-10)

    with tf.GradientTape() as gen_tape:
        latent = tf.random.normal(shape=[batch_size, latent_dim])
        fake_output = gan(latent)
        gen_loss = - tf.reduce_mean(fake_output)
    r_g = tf.abs(gen_loss - prev_gen_loss) / (prev_gen_loss + 1e-10)

    if (prev_dis_loss == 0 and prev_gen_loss == 0) or r_d > r_g * gamma:
        gan.discriminator.trainable = True
        dis_grad = dis_tape.gradient(dis_loss, gan.discriminator.trainable_variables)
        gan.discriminator.optimizer.apply_gradients(zip(dis_grad, gan.discriminator.trainable_variables))
    else:
        gan.discriminator.trainable = False
        gen_grad = gen_tape.gradient(gen_loss, gan.trainable_variables)
        gan.optimizer.apply_gradients(zip(gen_grad, gan.trainable_variables))

    return dis_loss, gen_loss
