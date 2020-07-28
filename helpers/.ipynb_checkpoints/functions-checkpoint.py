import tensorflow.keras.backend as K
import torch
import torch.nn.functional as F


def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true * y_pred)


def vae_loss(recon_x, x, mu, logvar):
	bce = F.binary_cross_entropy(recon_x, x, size_average=False)

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	kld = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
	return bce + kld