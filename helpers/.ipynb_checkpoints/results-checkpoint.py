# (Entire file) From Oriol - no longer maintained

import matplotlib.pyplot as plt
# from cfg.GAN_cfg import *
# from cfg.VAE_cfg import *
from cfg.GAN_cfg import cfg
import pandas as pd
from os import path
import numpy as np
import math
from contextlib import redirect_stdout


# Resahpe energy grids
def reshape_energy_grids(energy_grid, new_shape=30):
	reshaped_grid = energy_grid.reshape((30, 30))
	return reshaped_grid

#Create clusters of the energy grid to create an easy-to-read 3d bar plot
def create_cluster(energy_grid, num_of_cells):
	rows_0 = len(energy_grid)
	columns_0 = len(energy_grid[0])
	# print("original sizes: ", rows_0, columns_0)
	cells_dim = int(rows_0 / math.sqrt(num_of_cells))
	# print("cells_dim: ", cells_dim)
	columns_1 = int(columns_0 / cells_dim)
	rows_1 = int(rows_0 / cells_dim)

	new_grid = np.zeros((rows_1, columns_1))

	for i in range(rows_1):
		for j in range(columns_1):
			cluster = energy_grid[int(i * cells_dim):int((i + 1) * cells_dim - 1),
					  int(j * cells_dim):int((j + 1) * cells_dim - 1)]
			new_grid[i][j] = sum(sum(cluster)) / len(cluster)

	return new_grid


# Function to plot rowsXcolumns images from startiing point rand.
def plot_energy_grids(energy_grids_to_plot, rows=2, columns=2, name=""):
    fig=plt.figure(figsize=(12, 12))
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(energy_grids_to_plot[i].reshape((energy_deposit_input_shape[0], energy_deposit_input_shape[1])))
    plt.savefig(outputs_path + "images/" + name)
    #plt.show()


def save_model(model, name = "model"):
    # serialize model to JSON
    model_json = model.to_json()
    with open(outputs_path + "models/" + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(outputs_path + "models/" + name + ".h5")
    print("Saved model " + str(name) + " to disk")
def plot_energy_grid(energy_grid_to_plot, rows=2, columns=2):
	fig = plt.figure(figsize=(12, 12))
	for i in range(1, columns * rows + 1):
		plt.imshow(energy_grid_to_plot)
	plt.show()

# Plot 3d bar plot based on 2d bar plot
def plot_3d_bar(outputs_path, data, cells, name):
	# setup the figure and axes
	fig = plt.figure(figsize=(32, 12))
	ax1 = fig.add_subplot(121, projection='3d')

	# get the coordinates
	x = list(range(0, cells))
	y = list(range(0, cells))
	_xx, _yy = np.meshgrid(x, y)
	x, y = _xx.ravel(), _yy.ravel()

	# Get top & down values
	top = data.ravel()
	bottom = np.zeros_like(top)
	width = depth = 1

	ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
	plt.savefig(outputs_path + "images/" + name)

	plt.show()

# Function to plot rowsXcolumns images from startiing point rand.
def plot_energy_grids(outputs_path, energy_grids_to_plot, rows=2, columns=2, name=""):
	fig=plt.figure(figsize=(12, 12))
	for i in range(1, columns*rows +1):
		fig.add_subplot(rows, columns, i)
		plt.imshow(energy_grids_to_plot[i].reshape((30, 30)))
	#plt.show()
	plt.savefig(outputs_path + "images/" + name)
	#plt.savefig("C:/Users/uri_9/Desktop/Master/" + name)


def save_model(outputs_path, model, name="model"):
	# serialize model to JSON
	model_json = model.to_json()
	with open(outputs_path + "models/" + name + ".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	#model.save_weights(outputs_path + "models/" + name + ".h5")
	print("Saved model " + str(name) + " to disk")


def save_experiment_conf(name, model_id, g_train_history, d_train_history, g_test_history, d_test_history):

    if not path.exists(outputs_path + "cfg_data.xlsx"):

        generator_params = ['generator_n_epochs', 'generator_batch_size', 'generator_lr', 'generator_beta_1',
                             'generator_loss_func', 'generator test loss', 'generator test accuracy']
        dicscriminator_params = ['discriminator_n_epochs', 'discriminator_batch_size', 'discriminator_lr',
                                  'discriminator_beta_1', 'discriminator_loss_func', 'discriminator_metrics',
                                 'discriminator test loss', 'discriminator test accuracy']
        dataset_params = ['model_id', 'test_size', 'random_state']
        cfg_data = pd.DataFrame(columns = ['Name'] + dataset_params + generator_params + dicscriminator_params)

    else:
        cfg_data = pd.read_excel(outputs_path + "cfg_data.xlsx")
        cfg_data = cfg_data.drop(['Unnamed: 0'], axis=1)

    data = [name, model_id, test_size, random_state, g_n_epochs, g_batch_size, g_lr,
            g_beta_1, g_loss_func, g_test_history[0], g_test_history[1], d_n_epochs, d_batch_size, d_lr,
            d_beta_1, d_loss_func, d_metrics, d_test_history[0], d_test_history[1]]
    cfg_data.loc[name] = data
    cfg_data.to_excel(outputs_path + "cfg_data.xlsx")

def generate_metrics(gan_model, g_model, d_model, model_id, g_train_history, d_train_history, g_test_history, d_test_history):

	model_name = str(model_id) + "_gan_fold"
	save_model(d_model, name=str(model_id) + "_discriminator_fold")
	save_model(g_model, name=str(model_id) + "_generator_fold")
	save_model(gan_model, name=model_name)
	save_experiment_conf(model_name, model_id, g_train_history, d_train_history, g_test_history, d_test_history)
	if not path.exists(cfg["Global"]["Data"]["outputs_path"] + "cfg_data.xlsx"):

		generator_params = ['generator_epochs', 'generator_batch_size', 'generator_lr', 'generator_beta_1',
							 'generator_loss_func', 'generator test loss', 'generator test accuracy']
		dicscriminator_params = ['discriminator_epochs', 'discriminator_batch_size', 'discriminator_lr',
								  'discriminator_beta_1', 'discriminator_loss_func', 'discriminator_metrics',
								 'discriminator test loss', 'discriminator test accuracy']
		dataset_params = ['model_id', 'test_size', 'random_state']
		cfg_data = pd.DataFrame(columns = ['Name'] + dataset_params + generator_params + dicscriminator_params)

	else:
		cfg_data = pd.read_excel(cfg["Global"]["Data"]["outputs_path"] + "cfg_data.xlsx")
		cfg_data = cfg_data.drop(['Unnamed: 0'], axis=1)

	data = [name, model_id, cfg["Global"]["Data"]["test_size"], cfg["Global"]["Data"]["random_state"],
			cfg["Generator"]["Training"]["epochs"], cfg["Generator"]["Training"]["batch_size"],
			cfg["Generator"]["Training"]["lr"], cfg["Generator"]["Training"]["beta_1"],
			cfg["Generator"]["Training"]["loss_func"],  g_test_history[0], g_test_history[1],
			cfg["Discriminator"]["Training"]["epochs"],
			cfg["Generator"]["Training"]["loss_func"], g_test_history[0], g_test_history[1],
			cfg["Discriminator"]["Training"]["epochs"], cfg["Discriminator"]["Training"]["batch_size"],
			cfg["Discriminator"]["Training"]["lr"], cfg["Discriminator"]["Training"]["beta_1"],
			cfg["Discriminator"]["Training"]["loss_func"],
			cfg["Discriminator"]["Training"]["metrics"], d_test_history[0], d_test_history[1]]
	cfg_data.loc[name] = data
	cfg_data.to_excel(cfg["Global"]["Data"]["outputs_path"] + "cfg_data.xlsx")

def generate_metrics(outputs_path, gan_model, g_model, d_model, model_id, g_train_history, d_train_history, g_test_history, d_test_history):

	save_model(outputs_path, d_model, name="discriminator_fold")
	save_model(outputs_path, g_model, name="generator_fold")
	save_model(outputs_path, gan_model, name="gan_fold")
	save_architecture(outputs_path, d_model, model_name="discriminator_fold")
	save_architecture(outputs_path, g_model, model_name="generator_fold")
	save_architecture(outputs_path, gan_model, model_name="gan_fold")
	#save_experiment_conf(model_name, model_id, g_train_history, d_train_history, g_test_history, d_test_history)

def myprint(s):
    with open('./outputs/modelsummary.txt', 'w+') as f:
        print(s, file=f)



# =================
# Results visualization
# Credits for original visualization code: https://keras.io/examples/variational_autoencoder_deconv/
# (François Chollet).
# Adapted to accomodate this VAE.
# =================
def viz_latent_space(encoder, data):
    input_data, target_data = data
    mu, _, _ = encoder.predict(input_data)
    plt.figure(figsize=(8, 10))
    plt.scatter(mu[:, 0], mu[:, 1], c=target_data)
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    plt.colorbar()
    plt.show()

def viz_decoded(encoder, decoder, data):
    num_samples = 15
    figure = np.zeros((img_width * num_samples, img_height * num_samples, num_channels))
    grid_x = np.linspace(-4, 4, num_samples)
    grid_y = np.linspace(-4, 4, num_samples)[::-1]
    for i, yi in enumerate(grid_y):
      for j, xi in enumerate(grid_x):
          z_sample = np.array([[xi, yi]])
          x_decoded = decoder.predict(z_sample)
          digit = x_decoded[0].reshape(img_width, img_height, num_channels)
          figure[i * img_width: (i + 1) * img_width,
                  j * img_height: (j + 1) * img_height] = digit
    plt.figure(figsize=(10, 10))
    start_range = img_width // 2
    end_range = num_samples * img_width + start_range + 1
    pixel_range = np.arange(start_range, end_range, img_width)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
    # So reshape if necessary
    fig_shape = np.shape(figure)
    if fig_shape[2] == 1:
        figure = figure.reshape((fig_shape[0], fig_shape[1]))
    # Show image
    plt.imshow(figure)
    plt.show()
def save_architecture(outputs_path, model, model_name):

	with open(outputs_path + model_name + '.txt', 'w') as f:
		with redirect_stdout(f):
			model.summary()
