from sklearn.model_selection import train_test_split
from helpers.results import *
from cfg.GAN_cfg import *
from cfg.VAE_cfg import *
import numpy as np
import pandas as pd
from helpers.results import plot_energy_grids
from helpers.utilities import transform, inv_transform
from cfg.GAN_cfg import cfg
import numpy as np
import pandas as pd
import os
import json

import torch
from torch.utils.data import DataLoader

# This imports settings to be used globally, such as file path to our data, and other hyperparameters that might determine how models are built/trained
gan_params = GAN_config()
vae_params = VAE_config()

#################
# Discriminator #
# From Oriol - no longer maintained
#################
#Resahpe energy grids
def reshape_energy_grids(energy_grids):
    reshaped_grids = energy_grids.copy()
    for i, val in enumerate(reshaped_grids):
        reshaped_grids[i] = val.reshape((cfg["Discriminator"]["Data"]["energy_deposit_input_shape"][0], cfg["Discriminator"]["Data"]["energy_deposit_input_shape"][1]))
    return reshaped_grids

# Generate labels
def discriminator_input():
	data = pd.read_hdf(gan_params.df_path)
	data = pd.read_hdf(cfg["Global"]["Data"]["df_path"])
	energy_grids = data.EnergyDeposit.tolist()
	energy_grids = reshape_energy_grids(energy_grids)
	return np.asarray(energy_grids)


#############
# Generator #
# From Oriol - no longer maintained
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
# From Oriol - no longer maintained
#############
def load_regressor_input_data():
	g_input, discriminator_input = generate_input_data()
	return discriminator_input

def load_regressor_output_data():
	data = pd.read_hdf(gan_params.df_path)
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
# From Oriol - no longer maintained
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
# From Oriol - no longer maintained
##########
# Return x and y data
def generate_input_data():
	data = pd.read_hdf(vae_params.df_path)
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
# From Oriol - no longer maintained
def format_images(image):
	image = np.expand_dims(image, axis=-1)
	# convert from unsigned ints to floats
	image = image.astype('float32')
	# scale from [0,255] to [0,1]
	#image = image / 255.0

	return image

# load and prepare mnist training images
# From Oriol - no longer maintained
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


# ===========================
# Will's dataloader code 
# ===========================

class get_train_val_loaders:
    def __init__(self, batch_size, device):
        '''
        Args - 
            batch_size: int, the number of samples per batch
            device: 'cuda:0', 'cpu', the device to keep the data on. Data will not be transfered between devices
                on a per-batch basis, rather the entire dataset will be kept on this device and indexed to form batches 
                
        Returns - 
            Nothing, this simply CONSTRUCTS the dataset, which can be sampled with other functions 
            
        Usage -
        
            dataset = get_train_val_loaders()
            train_batch = dataset.get_train_batch()
            val_batch = dataset.get_val_batch()
        Notes - 
            1. Set the dataset path in the configuration file of this repository (parameter: sleepy_path_to_calo_ml)
            2. if you want to do any sort of normalization, it is better to do it in the model itself.
                that way if we want to visualize data, it will not be "messed with" if we extract it directly 
                from the dataloader. also, the model file will get logged during training with mlflow, so there
                is reason to keep it there in the first place 
            3. The number of events is hardcoded to 50,000
            4. The dataset is cycled automatically, so you can sample 40,000 batches of size 128 if you wanted, from the 50k total events
            5. this is great template code for any future dataloader. It does not really get any faster than this.
        '''
        self.batch_size = batch_size
        self.device = device
        spacal_df = pd.read_hdf(vae_params.sleepy_path_to_calo_ml)

        # make a copy for normalization, convert to numpy
        spacal_series = spacal_df['EnergyDeposit']
        spacal_array = np.array(spacal_series.tolist())
        n_spacal_array = spacal_array
        momentum = torch.Tensor(spacal_df['ParticleMomentum'].apply(list).tolist())
#         momentum -= momentum.mean()
#         momentum /= momentum.std()
        position = torch.Tensor(spacal_df['ParticlePoint'].apply(list).tolist())[:, :2]
#         position /= position.std()

#         everything = torch.cat((position, momentum), dim=1)

        # new log scaling code, arbitrary
        n_spacal_array = transform(n_spacal_array)

        # needs to be padded to 32x32 so we can donwsample twice. (three times?) lmao i have no idea
        padded_array = np.pad(n_spacal_array.reshape(n_spacal_array.shape[0], 30, 30), 1)[1:-1]

        # since we are treating this like an image, we have to unsqueeze at dim 1 to make it in the dimension of a
        # greyscale image. i.e. it expects typically either (3, x, y) or (1, x, y), we currently have (x, y)
        train_array = torch.Tensor(padded_array[round(50000*vae_params.validation_split):]).unsqueeze(1)
        # we want to learn rotational symmetry, so it is a good idea to do data augmentation by means of 90, 180, and 270 degree rotations 
        train_90 = torch.rot90(train_array, 1, [-2, -1])
        train_180 = torch.rot90(train_90, 1, [-2, -1])
        train_270 = torch.rot90(train_180, 1, [-2, -1])
        self.train_array = torch.cat([train_array, train_90, train_180, train_270], dim=0)
        self.val_array = torch.Tensor(padded_array[:round(50000*vae_params.validation_split)]).unsqueeze(1)
        self.train_momentum = torch.cat(4*[momentum[round(50000*vae_params.validation_split):]], dim=0)
        self.train_positions = torch.cat(4*[position[round(50000*vae_params.validation_split):]], dim=0)
#         self.train_attributes = momentum[round(50000*vae_params.validation_split):]
        self.val_momentum = momentum[:round(50000*vae_params.validation_split)]
        self.val_positions = position[:round(50000*vae_params.validation_split)]
        
        self.train_length = len(self.train_array)
        self.val_length = len(self.val_array)
        print(self.train_array.shape)
        print(self.val_array.shape)
        self.batches = 0

    def __len__(self):
        return len(self.train_array)
    
    def get_train_batch(self):
        remaining_batches = -(-self.train_length // self.batch_size)
        if self.batches == remaining_batches-1:
            self.batches = 0
        images = self.train_array[self.batches*self.batch_size:(self.batches+1)*self.batch_size]
        momentum = self.train_momentum[self.batches*self.batch_size:(self.batches+1)*self.batch_size]
        positions = self.train_positions[self.batches*self.batch_size:(self.batches+1)*self.batch_size]
        self.batches += 1
        return images.to(self.device), momentum.to(self.device), positions.to(self.device)
    
    def get_val_batch(self):
        remaining_batches = -(-self.val_length // self.batch_size)
        if self.batches == remaining_batches-1:
            self.batches = 0
        images = self.val_array[self.batches*self.batch_size:(self.batches+1)*self.batch_size]
        momentum = self.val_momentum[self.batches*self.batch_size:(self.batches+1)*self.batch_size]
        positions = self.val_positions[self.batches*self.batch_size:(self.batches+1)*self.batch_size]
        self.batches += 1
        return images.to(self.device), momentum.to(self.device), positions.to(self.device)
    
def set_up_data():
# From Oriol - no longer maintained
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
