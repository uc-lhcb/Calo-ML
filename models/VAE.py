from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
from cfg.VAE_cfg import *


class VAE:

    def __init__(self, test):

        print("VAE initialized: ", str(test))

    # # =================
    # # Encoder
    # # =================

    def define_autoencoder(self):
        # Definition
        self.i       = Input(shape=input_shape, name='encoder_input')
        cx      = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(self.i)
        cx      = BatchNormalization()(cx)
        x       = Flatten()(cx)
        x       = Dense(20, activation='relu')(x)
        x       = BatchNormalization()(x)
        self.mu      = Dense(latent_dim, name='latent_self.mu')(x)
        self.sigma   = Dense(latent_dim, name='latent_self.sigma')(x)

        # Get Conv2D shape for Conv2DTranspose operation in decoder
        self.conv_shape = K.int_shape(cx)

        # Define sampling with reparameterization trick
        def sample_z(args):
          self.mu, self.sigma = args
          batch     = K.shape(self.mu)[0]
          dim       = K.int_shape(self.mu)[1]
          eps       = K.random_normal(shape=(batch, dim))
          return self.mu + K.exp(self.sigma / 2) * eps

        # Use reparameterization trick to ....??
        z       = Lambda(sample_z, output_shape=(latent_dim, ), name='z')([self.mu, self.sigma])

        # Instantiate encoder
        self.encoder = Model(self.i, [self.mu, self.sigma, z], name='encoder')
        self.encoder.summary()
        return self.encoder

    # =================
    # Decoder
    # =================

    def define_decoder(self):
        d_i   = Input(shape=(latent_dim, ), name='decoder_input')
        x     = Dense(self.conv_shape[1] * self.conv_shape[2] * self.conv_shape[3], activation='relu')(d_i)
        x     = BatchNormalization()(x)
        x     = Reshape((self.conv_shape[1], self.conv_shape[2], self.conv_shape[3]))(x)
        cx    = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        cx    = BatchNormalization()(cx)
        o     = Conv2DTranspose(filters=num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)

        # Instantiate decoder
        self.decoder = Model(d_i, o, name='decoder')
        self.decoder.summary()

        return self.decoder

    # =================
    # VAE as a whole
    # =================

    # Instantiate VAE
    def create_vae(self):
        vae_outputs = self.decoder(self.encoder(self.i)[2])
        vae         = Model(self.i, vae_outputs, name='vae')
        vae.summary()

        # Define loss
        def kl_reconstruction_loss(true, pred):
            # Reconstruction loss
            reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
            # KL divergence loss
            kl_loss = 1 + self.sigma - K.square(self.mu) - K.exp(self.sigma)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            # Total loss = 50% rec + 50% KL divergence loss
            return K.mean(reconstruction_loss + kl_loss)

        # Compile VAE
        vae.compile(optimizer='adam', loss=kl_reconstruction_loss)

        return vae
