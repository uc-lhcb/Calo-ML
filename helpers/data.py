from sklearn.model_selection import train_test_split
from helpers.results import *
from cfg.GAN_cfg import *
from cfg.VAE_cfg import *
import numpy as np
import pandas as pd

# i do not like it either
import torch
from torch.utils.data import DataLoader

gan_params = GAN_config()
vae_params = VAE_config()

#################
# Discriminator #
#################
#Resahpe energy grids
def reshape_energy_grids(energy_grids):
    reshaped_grids = energy_grids.copy()
    for i, val in enumerate(reshaped_grids):
        reshaped_grids[i] = val.reshape((energy_deposit_input_shape[0], energy_deposit_input_shape[1]))
    return reshaped_grids

# Generate labels
def discriminator_input():
	data = pd.read_hdf(gan_params.df_path)
	energy_grids = data.EnergyDeposit.tolist()
	energy_grids = reshape_energy_grids(energy_grids)
	return np.asarray(energy_grids)


#############
# Generator #
#############
# Generate X
def generate_noise(data, length):
	# Generate noise for each example we have
	if noise_dist == "normalized_gaussian":
		mu = 0
		sigma = 1
		noise = []
		for i in range(data.shape[0]):
			noise.append(np.random.normal(mu, sigma, length).tolist())
	return np.asarray(noise)

# Generate X data
def load_x(data):
	params = []
	for x_param in g_input_parameters:
		if x_param == "ParticlePoint":
			params.append(np.array(data[str(x_param)].to_list())[:, :-1])
		else:
			params.append(np.array(data[str(x_param)].to_list())[:, :])

	return params

#############
# Regressor #
#############
def load_regressor_input_data():
	g_input, discriminator_input = generate_input_data()
	return discriminator_input

def load_regressor_output_data():
	data = pd.read_hdf(gan_params.df_path)
	output_data = load_x(data)
	"""
	for i, val in enumerate(output_data):
		print("number: ", i)
		print("Type: ", type(val))
		output_data[i] = np.array(val)
		print("Type: ", type(output_data[i]))
	"""

	output_data = np.concatenate(output_data, axis=1)

	return output_data

def load_regressor_samples():

	r_input_data = load_regressor_input_data()
	r_train_input_data ,r_test_input_data = train_test_split(r_input_data, test_size=test_size, random_state=random_state)
	r_train_input_data = format_images(r_train_input_data)
	r_test_input_data = format_images(r_test_input_data)

	r_output_data = load_regressor_output_data()
	r_train_output_data, r_test_output_data = train_test_split(r_output_data, test_size=test_size, random_state=random_state)
	r_train_output_data = np.transpose(r_train_output_data).tolist()
	r_test_output_data = np.transpose(r_test_output_data).tolist()

	return r_train_input_data, r_train_output_data, r_test_input_data, r_test_output_data


##########
# Global #
##########
# Return x and y data
def generate_input_data():
	data = pd.read_hdf(vae_params.df_path)

	if add_g_input_parameters:
		input_data = load_x(data)
		input_data = np.concatenate(input_data, axis=1)
		length = g_input_length - len(input_data[0])
		noise = generate_noise(input_data, length)
		half_length = int(len(np.asarray(noise))/2)
		g_input = np.concatenate((np.asarray(noise)[:half_length], input_data, np.asarray(noise)[half_length:]), axis=1)
	else:
		g_input = generate_noise(data, g_input_length)

	return g_input, discriminator_input()

# Fromat images
def format_images(image):
	image = np.expand_dims(image, axis=-1)
	# convert from unsigned ints to floats
	image = image.astype('float32')
	# scale from [0,255] to [0,1]
	#image = image / 255.0

	return image

# load and prepare mnist training images
def load_real_samples(model_id):

	g_input_data, d_input_data = generate_input_data()
	plot_energy_grids(d_input_data, name=str(model_id) + "_real_images")
	train_g_input_data, test_g_input_data, train_d_input_data, test_d_input_data = \
		train_test_split(g_input_data, d_input_data, test_size=test_size, random_state=random_state)

	train_d_input_data = format_images(train_d_input_data)
	test_d_input_data = format_images(test_d_input_data)
	train_g_input_data = train_g_input_data.reshape(train_d_input_data.shape[0], g_input_length)
	test_g_input_data = test_g_input_data.reshape(test_d_input_data.shape[0], g_input_length)

	return train_g_input_data, test_g_input_data, train_d_input_data, test_d_input_data


# Will's dataloader code 
def get_train_val_loaders():
	spacal_df = pd.read_hdf(vae_params.df_path)

	# make a copy for normalization, convert to numpy
	spacal_series = spacal_df['EnergyDeposit']
	spacal_array = np.array(spacal_series.tolist())
	n_spacal_array = spacal_array

	# maybe delete later, trying to normalize for some reason
	n_spacal_array -= n_spacal_array.mean(1)[:, np.newaxis]
	n_spacal_array /= n_spacal_array.std(1)[:, np.newaxis]

	# needs to be padded to 32x32 so we can donwsample twice. (three times?) lmao i have no idea
	padded_array = np.pad(n_spacal_array.reshape(n_spacal_array.shape[0], 30, 30), 1)[1:-1]

	# since we are treating this like an image, we have to unsqueeze at dim 1 to make it in the dimension of a
	# greyscale image. i.e. it expects typically either (3, x, y) or (1, x, y), we currently have (x, y)
	train_array = torch.Tensor(padded_array[round(50000*vae_params.validation_split):]).unsqueeze(1)
	val_array = torch.Tensor(padded_array[:round(50000*vae_params.validation_split)]).unsqueeze(1)

	# TODO(will) see if pin_memory = True is faster. also, see if a dataset is faster as opposed to just a big tensor
	train_loader = DataLoader(train_array, batch_size=vae_params.batch_size, shuffle=True)
	val_loader = DataLoader(val_array, batch_size=vae_params.batch_size, shuffle=True)

	return train_loader, val_loader
