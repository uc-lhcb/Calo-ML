from sklearn.model_selection import train_test_split
from helpers.results import plot_energy_grids
from cfg.GAN_cfg import cfg
import numpy as np
import pandas as pd
import os
import json


#################
# Discriminator #
#################
#Resahpe energy grids
def reshape_energy_grids(energy_grids):
    reshaped_grids = energy_grids.copy()
    for i, val in enumerate(reshaped_grids):
        reshaped_grids[i] = val.reshape((cfg["Discriminator"]["Data"]["energy_deposit_input_shape"][0], cfg["Discriminator"]["Data"]["energy_deposit_input_shape"][1]))
    return reshaped_grids

# Generate labels
def discriminator_input():
	data = pd.read_hdf(cfg["Global"]["Data"]["df_path"])
	energy_grids = data.EnergyDeposit.tolist()
	energy_grids = reshape_energy_grids(energy_grids)
	return np.asarray(energy_grids)


#############
# Generator #
#############
# Generate X
def generate_noise(data, length):
	# Generate noise for each example we have
	if cfg["Generator"]["Data"]["noise_dist"] == "normalized_gaussian":
		mu = 0
		sigma = 1
		noise = []
		for i in range(data.shape[0]):
			noise.append(np.random.normal(mu, sigma, length).tolist())
	return np.asarray(noise)

# Generate X data
def load_x(data):
	params = []
	for x_param in cfg["Generator"]["Data"]["input_parameters"]:
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
	data = pd.read_hdf(cfg["Global"]["Data"]["df_path"])
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
	r_train_input_data ,r_test_input_data = train_test_split(r_input_data, test_size=cfg["Global"]["Data"]["test_size"], random_state=cfg["cfg_global"]["Data"]["random_state"])
	r_train_input_data = format_images(r_train_input_data)
	r_test_input_data = format_images(r_test_input_data)

	r_output_data = load_regressor_output_data()
	r_train_output_data, r_test_output_data = train_test_split(r_output_data, test_size=cfg["Global"]["Data"]["test_size"], random_state=cfg["cfg_global"]["Data"]["random_state"])
	r_train_output_data = np.transpose(r_train_output_data).tolist()
	r_test_output_data = np.transpose(r_test_output_data).tolist()

	return r_train_input_data, r_train_output_data, r_test_input_data, r_test_output_data


##########
# Global #
##########
# Return x and y data
def generate_input_data():
	data = pd.read_hdf(cfg["Global"]["Data"]["df_path"])

	if cfg["Generator"]["Data"]["add_input_parameters"]:
		input_data = load_x(data)
		input_data = np.concatenate(input_data, axis=1)
		length = cfg["Generator"]["Data"]["input_length"] - len(input_data[0])
		noise = generate_noise(input_data, length)
		half_length = int(len(np.asarray(noise))/2)
		g_input = np.concatenate((np.asarray(noise)[:half_length], input_data, np.asarray(noise)[half_length:]), axis=1)
	else:
		g_input = generate_noise(data, cfg["Generator"]["Data"]["input_length"])

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
def load_real_samples(outputs_path):

	g_input_data, d_input_data = generate_input_data()
	plot_energy_grids(outputs_path, d_input_data, name="real_images")
	train_g_input_data, test_g_input_data, train_d_input_data, test_d_input_data = \
		train_test_split(g_input_data, d_input_data, test_size=cfg["Global"]["Data"]["test_size"],
						 random_state=cfg["Global"]["Data"]["random_state"])

	train_d_input_data = format_images(train_d_input_data)
	test_d_input_data = format_images(test_d_input_data)
	train_g_input_data = train_g_input_data.reshape(train_d_input_data.shape[0], cfg["Generator"]["Data"]["input_length"])
	test_g_input_data = test_g_input_data.reshape(test_d_input_data.shape[0], cfg["Generator"]["Data"]["input_length"])

	return train_g_input_data, test_g_input_data, train_d_input_data, test_d_input_data

def set_up_data():
	# Read model_id & updated it
	with open(cfg["Global"]["Data"]["model_id_path"], "r") as f:
		model_id = int(f.readlines()[0])
		f.close()

	outputs_path = cfg["Global"]["Data"]["outputs_path"] + cfg["Global"]["Training"]["type"] + "_" + str(model_id) + "/"
	if not os.path.exists(outputs_path):
		os.makedirs(outputs_path)
		os.makedirs(outputs_path + "/images/")
		os.makedirs(outputs_path + "/models/")

	with open(outputs_path + "cfg.json", "w") as f:
		json.dump(cfg, f)

	return model_id, outputs_path