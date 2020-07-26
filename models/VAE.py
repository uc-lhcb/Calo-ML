from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from .cfg.VAE_cfg import *


params = VAE_config()

class VAE:

    def __init__(self, test):

        print("VAE initialized: ", str(test))

    # # =================
    # # Encoder
    # # =================

    def define_autoencoder(self):
        # Definition
        self.i       = Input(shape=params.input_shape, name='encoder_input')
        cx      = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(self.i)
        cx      = BatchNormalization()(cx)
        x       = Flatten()(cx)
        x       = Dense(20, activation='relu')(x)
        x       = BatchNormalization()(x)
        self.mu      = Dense(params.latent_dim, name='latent_self.mu')(x)
        self.sigma   = Dense(params.latent_dim, name='latent_self.sigma')(x)

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
        z       = Lambda(sample_z, output_shape=(params.latent_dim, ), name='z')([self.mu, self.sigma])

        # Instantiate encoder
        self.encoder = Model(self.i, [self.mu, self.sigma, z], name='encoder')
        self.encoder.summary()
        return self.encoder

    # =================
    # Decoder
    # =================

    def define_decoder(self):
        d_i   = Input(shape=(params.latent_dim, ), name='decoder_input')
        x     = Dense(self.conv_shape[1] * self.conv_shape[2] * self.conv_shape[3], activation='relu')(d_i)
        x     = BatchNormalization()(x)
        x     = Reshape((self.conv_shape[1], self.conv_shape[2], self.conv_shape[3]))(x)
        cx    = Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(x)
        cx    = BatchNormalization()(cx)
        o     = Conv2DTranspose(filters=params.num_channels, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(cx)

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


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False, upsample=False):
        super().__init__()
        self.convrelu = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.ReLU()]
        if downsample:
            self.convrelu.append(nn.MaxPool2d(2))

        if upsample:
            self.convrelu.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2))

    def forward(self, x):
        return nn.Sequential(*self.convrelu)(x)

# ==================
# Will's VAE code
# ==================
from torch import nn

class VAE(nn.Module):
    def __init__(self, n, z_dim):
        super().__init__()
        self.n = n

        self.encoder = nn.Sequential(
            ConvRelu(1, n),
            ConvRelu(n, n, downsample=True),
            ConvRelu(n, n),
            ConvRelu(n, n, downsample=True),
            ConvRelu(n, n // 2),
            nn.Flatten())

        h_dim = (n // 2) * (32 // 4) * (32 // 4)
        self.mu_linear = nn.Linear(h_dim, z_dim)
        self.logvar_linear = nn.Linear(h_dim, z_dim)
        self.z_linear = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            ConvRelu(n // 2, n),
            ConvRelu(n, n, upsample=True),
            ConvRelu(n, n),
            ConvRelu(n, n, upsample=True),
            ConvRelu(n, 1))

    def forward(self, x):
        encoded_image = self.encoder(x)

        z, mu, logvar = self.bottleneck(encoded_image)
        z = self.z_linear(z)
        z = z.view(z.size(0), self.n // 2, self.n // 4, self.n // 4)
        decoded_img = self.decoder(z)

        return decoded_img, mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.mu_linear(h), self.logvar_linear(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


# ==================
# Will's VQVAE code
# ==================


import torch
from torch import nn
import torch.nn.functional as F


class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * i.sigmoid()
        ctx.save_for_backward(result, i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, i = ctx.saved_variables
        sigmoid_x = i.sigmoid()
        return grad_output * (result + sigmoid_x * (1 - result))


swish = Swish.apply


class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            torch.sum(embed_onehot_sum)
            torch.sum(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=3, extra_layers=1, residual=True):
        super().__init__()
        self.residual = residual

        layers = [
            nn.Conv2d(in_channel, out_channel, stride=stride, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(out_channel),
            #             nn.ReLU()]
            Swish_module()]

        extra_block = [
            nn.Conv2d(out_channel, out_channel, stride=1, kernel_size=3, padding=(3 - 1) // 2),
            nn.BatchNorm2d(out_channel),
            #             nn.ReLU()]
            Swish_module()]

        layers.extend(extra_block)

        self.resblock = nn.Sequential(*layers)

    def forward(self, input):
        if self.residual:
            out = self.resblock(input)
            out = input + out
            return out
        else:
            out = self.resblock(input)
            return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, extra_layers, stride, kernel_size, residual, extra_residual_blocks,
                 downsample):
        super().__init__()

        self.out_channels = channel

        blocks = [
            ResBlock(in_channel, channel, extra_layers=extra_layers, stride=stride, residual=residual),
            Swish_module()
            #             nn.ReLU(inplace=True)
        ]

        for i in range(extra_residual_blocks):
            blocks.append(ResBlock(in_channel=channel, out_channel=channel, extra_layers=extra_layers, residual=True))
            if (downsample == 'Once') & (i == 0):
                blocks.append(nn.MaxPool2d(2, 2))
            if (downsample == 'Twice') & ((i == 0) | (i == 1)):
                blocks.append(nn.MaxPool2d(2, 2))

        self.encode = nn.Sequential(*blocks)

    def forward(self, input):
        return self.encode(input)


class Decoder(nn.Module):
    def __init__(self, channel, out_channel, extra_layers, extra_residual_blocks, upsample):
        super().__init__()

        blocks = []

        for i in range(extra_residual_blocks):
            blocks.append(ResBlock(in_channel=channel, out_channel=channel, extra_layers=extra_layers, residual=True))
            if (upsample == 'Twice') & (i == 0):
                blocks.append(nn.ConvTranspose2d(channel, channel, 4, 2, padding=1))

        blocks.append(nn.ConvTranspose2d(channel, out_channel, 4, 2, padding=1))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    '''
    params: in_channel=3, channel=64, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99
    '''

    def __init__(
            self,
            in_channel=3,
            channel=128,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
            n_embed=512,
            decay=0.99
    ):
        '''
        params: in_channel=3, channel=64, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=512, decay=0.99
        '''
        super().__init__()
        # Encoders, first one should have two rounds of downsampling, second should have one
        self.enc_b = Encoder(in_channel=in_channel, channel=channel, extra_layers=2, stride=2, kernel_size=5,
                             residual=False, extra_residual_blocks=2, downsample='Once')
        self.enc_t = Encoder(in_channel=channel, channel=channel, extra_layers=3, stride=1, kernel_size=3,
                             residual=False, extra_residual_blocks=2, downsample='Once')

        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)

        # Decoders,
        #         self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, extra_residual_blocks=n_res_block, upsample='Once')
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(embed_dim, embed_dim, 4, stride=2, padding=1)
        #         self.dec = Decoder(embed_dim + embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.dec = Decoder(embed_dim + embed_dim, in_channel, extra_layers=2, extra_residual_blocks=2, upsample='Twice')

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


