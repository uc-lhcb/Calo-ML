import matplotlib.pyplot as plt
from cfg.GAN_cfg import cfg
import pandas as pd
from os import path
import numpy as np
import math

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
	plt.savefig(outputs_path + "images/" + name)
	#plt.show()


def save_model(outputs_path, model, name="model"):
	# serialize model to JSON
	model_json = model.to_json()
	with open(outputs_path + "models/" + name + ".json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(outputs_path + "models/" + name + ".h5")
	print("Saved model " + str(name) + " to disk")


def save_experiment_conf(name, model_id, g_train_history, d_train_history, g_test_history, d_test_history):

	if not path.exists(cfg["Global"]["Data"]["outputs_path"] + "cfg_data.xlsx"):

		generator_params = ['generator_n_epochs', 'generator_batch_size', 'generator_lr', 'generator_beta_1',
							 'generator_loss_func', 'generator test loss', 'generator test accuracy']
		dicscriminator_params = ['discriminator_n_epochs', 'discriminator_batch_size', 'discriminator_lr',
								  'discriminator_beta_1', 'discriminator_loss_func', 'discriminator_metrics',
								 'discriminator test loss', 'discriminator test accuracy']
		dataset_params = ['model_id', 'test_size', 'random_state']
		cfg_data = pd.DataFrame(columns = ['Name'] + dataset_params + generator_params + dicscriminator_params)

	else:
		cfg_data = pd.read_excel(cfg["Global"]["Data"]["outputs_path"] + "cfg_data.xlsx")
		cfg_data = cfg_data.drop(['Unnamed: 0'], axis=1)

	data = [name, model_id, cfg["Global"]["Data"]["test_size"], cfg["Global"]["Data"]["random_state"],
			cfg["Generator"]["Training"]["n_epochs"], cfg["Generator"]["Training"]["batch_size"],
			cfg["Generator"]["Training"]["lr"], cfg["Generator"]["Training"]["beta_1"],
			cfg["Generator"]["Training"]["loss_func"],  g_test_history[0], g_test_history[1],
			cfg["Discriminator"]["Training"]["n_epochs"],
			cfg["Generator"]["Training"]["loss_func"], g_test_history[0], g_test_history[1],
			cfg["Discriminator"]["Training"]["n_epochs"], cfg["Discriminator"]["Training"]["batch_size"],
			cfg["Discriminator"]["Training"]["lr"], cfg["Discriminator"]["Training"]["beta_1"],
			cfg["Discriminator"]["Training"]["loss_func"],
			cfg["Discriminator"]["Training"]["metrics"], d_test_history[0], d_test_history[1]]
	cfg_data.loc[name] = data
	cfg_data.to_excel(cfg["Global"]["Data"]["outputs_path"] + "cfg_data.xlsx")

def generate_metrics(outputs_path, gan_model, g_model, d_model, model_id, g_train_history, d_train_history, g_test_history, d_test_history):

	save_model(outputs_path, d_model, name="discriminator_fold")
	save_model(outputs_path, g_model, name="generator_fold")
	save_model(outputs_path, gan_model, name="gan_fold")
	#save_experiment_conf(model_name, model_id, g_train_history, d_train_history, g_test_history, d_test_history)
