
import keras
from keras import layers, models
from config import *

latent_dim = model_config.latent_dim
input_shape = model_config.image_shape

class Generator(models.Model):
    def __init__(self, latent_dim=latent_dim):
        x = layers.Input((latent_dim,))

        h = layers.Dense(7 * 7 * 256, kernel_initializer='he_normal')(x)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        h = layers.Reshape((7, 7, 256))(h)

        h = layers.Conv2DTranspose(256, 3, 1, padding='same', kernel_initializer='he_normal')(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        h = layers.Conv2DTranspose(128, 3, 2, padding='same', kernel_initializer='he_normal')(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        h = layers.Conv2DTranspose(128, 3, 1, padding='same', kernel_initializer='he_normal')(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        h = layers.Conv2DTranspose(64, 3, 2, padding='same', kernel_initializer='he_normal')(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        y = layers.Conv2DTranspose(1, 3, 1, padding='same', kernel_initializer='he_normal', activation='tanh')(h)

        super().__init__(x, y)


class Discriminator(models.Model):
    def __init__(self, input_shape=input_shape):
        x = layers.Input(input_shape)

        h = layers.Conv2D(64, 3, 1, padding='same', kernel_initializer='he_normal')(x)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        h = layers.Conv2D(128, 3, 2, padding='same', kernel_initializer='he_normal')(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        h = layers.Conv2D(128, 3, 1, padding='same', kernel_initializer='he_normal')(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        h = layers.Conv2D(256, 3, 2, padding='same', kernel_initializer='he_normal')(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)

        h = layers.Flatten()(h)

        h = layers.Dense(256, kernel_initializer='he_normal')(h)
        h = layers.BatchNormalization()(h)
        h = layers.LeakyReLU()(h)
        h = layers.Dropout(0.3)(h)

        y = layers.Dense(1)(h) # Non Saturating Gan

        super().__init__(x, y)
        self.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0., beta_2=0.9))


class GAN(models.Model):
    def __init__(self, latent_dim=latent_dim):
        discriminator = Discriminator()
        generator = Generator()

        gan_input = layers.Input((latent_dim))
        gan_output = discriminator(generator(gan_input))

        super().__init__(gan_input, gan_output)
        self.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0., beta_2=0.9))
        self.generator = generator
        self.discriminator = discriminator

