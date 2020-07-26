from results import *
from cfg.GAN_cfg import *
from cfg.VAE_cfg import *
import numpy as np

gan_params = GAN_config()
vae_params = VAE_config()


def test(g_model, d_model, gan_model, g_test_input_data, d_test_input_data, model_id):

	number_of_samples = d_test_input_data.shape[0]
	# get randomly selected 'real' samples
	x_real = d_test_input_data
	y_real = np.ones((number_of_samples, 1))
	# generate 'fake' examples
	# generate points in latent space
	X_fake = g_model.predict(g_test_input_data)

	y_fake = np.zeros((number_of_samples, 1))
	# create training set for the discriminator
	X, y = np.vstack((x_real, X_fake)), np.vstack((y_real, y_fake))
	# update discriminator model weights
	d_history = d_model.evaluate(X, y, verbose=2)
	print("discriminator history", d_history)
	# prepare points in latent space as input for the generator
	X_gan = g_test_input_data
	# create inverted labels for the fake samples
	y_gan = np.ones((number_of_samples, 1))
	# update the generator via the discriminator's error
	g_history = gan_model.evaluate(X_gan, y_gan, verbose = 2)

	plot_energy_grids(X_fake, name=str(model_id) + "_generated_images")

	print("generator history", g_history)

	return d_history, g_history

def train_vae(vae, input_train):

	# Train autoencoder
	vae_history = vae.fit(input_train, input_train, epochs=vae_params.no_epochs, batch_size=vae_params.batch_size, validation_split=vae_params.validation_split)

	return vae_history

def train(g_model, d_model, gan_model, g_train_input_data, d_train_input_data, model_id):

    number_of_samples = d_train_input_data.shape[0]
	# get randomly selected 'real' samples
	x_real = d_train_input_data
	y_real = np.ones((number_of_samples, 1))

	for i in range(gan_params.training_iterations):

		first_split_point = int((i)*number_of_samples/gan_params.training_iterations)
		second_split_point = int((i+1)*number_of_samples/gan_params.training_iterations)
		# generate 'fake' examples
		# generate points in latent space
		# x_input = generate_latent_points(latent_dim, number_of_samples)
		X_fake = g_model.predict(g_train_input_data[first_split_point:second_split_point])
		y_fake = np.zeros((second_split_point - first_split_point, 1))
		plot_energy_grids(X_fake, name=str(model_id) + "_fake_images_" + str(i))

		X, y = np.vstack((x_real[first_split_point:second_split_point], X_fake)),\
			   np.vstack((y_real[first_split_point:second_split_point], y_fake))

		# prepare points in latent space as input for the generator
		# X_gan = generate_latent_points(latent_dim, number_of_samples)
		X_gan = g_train_input_data[first_split_point:second_split_point]
		# create inverted labels for the fake samples
		y_gan = np.ones((second_split_point - first_split_point, 1))

		# update discriminator model weights
		d_history = d_model.fit(X, y, validation_split=gan_params.d_val_size, batch_size=gan_params.d_batch_size, epochs=gan_params.d_n_epochs, verbose=2)
		# update the generator via the discriminator's error
		g_history = gan_model.fit(X_gan, y_gan, validation_split=gan_params.g_val_size, batch_size=gan_params.g_batch_size, epochs=gan_params.g_n_epochs, verbose=2)

	return d_history, g_history

def train_r_model(r_model, train_input_data, train_output_data):

	r_history = r_model.fit(train_input_data, train_output_data, validation_split=r_val_size, batch_size=r_batch_size, epochs=r_n_epochs,
							  verbose=2)

	return r_history

def r_test(r_model, r_test_input_data, r_test_output_data):

	r_history = r_model.evaluate(r_test_input_data, r_test_output_data, verbose = 2)

	return r_history


import torch
from torch import nn, optim
from utilities import count_parameters
from torchvision import utils
from tqdm import tqdm

def train_VQVAE(epoch, loader, model, optimizer, device):
    '''
    params: epoch, loader, model, optimizer, device
    checkpoint gets saved to "run_stats.pyt"
    '''
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0
    params = count_parameters(model)
    for i, (img, labels) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                f"lr: {lr:.5f}"
            )
        )

        if i % 20 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),
                f"samples/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg",
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )

            torch.save({
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            }, 'run_stats.pyt')
            model.train()

        ret = {'Metric: Latent Loss':latent_loss.item(), 'Metric: Average MSE':mse_sum/mse_n, 'Metric: Reconstruction Loss':recon_loss.item(), 'Parameter: Parameters':params, 'Artifact':'run_stats.pyt'}
        yield ret


def train_VAE(epoch, loader, model, optimizer, device):
    '''
    params: epoch, loader, model, optimizer, device
    checkpoint gets saved to "run_stats.pyt"
    '''
    loader = tqdm(loader)

    loss_sum = 0
    params = count_parameters(model)
    for i, img in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        recon_image, mu, logvar = model(img)
        loss = vae_loss(recon_image, img, mu, logvar)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * img.shape[0]

        # lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"epoch: {epoch + 1}; loss: {loss.item():.5f}; "
            )
        )

        if i % 20 == 0:
            model.eval()
            sample_size = 10
            sample = img[:sample_size]

            with torch.no_grad():
                out, _ = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),
                f"samples/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg",
                nrow=sample_size,
                # normalize=True,
                # range=(-1, 1),
            )

            torch.save({
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            }, 'run_stats.pyt')
            model.train()

        ret = {'Metric: Loss':loss.item(), 'Parameter: Parameters':params, 'Artifact':'run_stats.pyt'}
        yield ret
