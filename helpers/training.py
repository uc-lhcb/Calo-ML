from helpers.results import plot_energy_grids
from cfg.GAN_cfg import cfg
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np


def test(outputs_path, g_model, d_model, gan_model, g_test_input_data, d_test_input_data, model_id):

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

	plot_energy_grids(outputs_path, X_fake, name="generated_images")

	print("generator history", g_history)

	return d_history, g_history


def train(outputs_path, g_model, d_model, gan_model, g_train_input_data, d_train_input_data, model_id):

	number_of_samples = d_train_input_data.shape[0]
	# get randomly selected 'real' samples
	x_real = d_train_input_data
	y_real = np.ones((number_of_samples, 1))

	for i in range(cfg["Global"]["Training"]["training_iterations"]):

		first_split_point = int((i)*number_of_samples/cfg["Global"]["Training"]["training_iterations"])
		second_split_point = int((i+1)*number_of_samples/cfg["Global"]["Training"]["training_iterations"])
		# generate 'fake' examples
		# generate points in latent space
		# x_input = generate_latent_points(latent_dim, number_of_samples)
		X_fake = g_model.predict(g_train_input_data[first_split_point:second_split_point])
		y_fake = np.zeros((second_split_point - first_split_point, 1))
		plot_energy_grids(outputs_path, X_fake, name="fake_images_" + str(i))

		X, y = np.vstack((x_real[first_split_point:second_split_point], X_fake)),\
			   np.vstack((y_real[first_split_point:second_split_point], y_fake))

		# prepare points in latent space as input for the generator
		# X_gan = generate_latent_points(latent_dim, number_of_samples)
		X_gan = g_train_input_data[first_split_point:second_split_point]
		# create inverted labels for the fake samples
		y_gan = np.ones((second_split_point - first_split_point, 1))

		"""
		# Code to tune hyperparameters
		keras_estimator = KerasClassifier(build_fn=define_discriminator, verbose=1)
		param_grid = dict(batch_size=d_batch_size_grid)
		grid = GridSearchCV(estimator=keras_estimator, param_grid=param_grid)
		grid.fit(X, y)
		print(grid.best_score_)
		print(grid.best_estimator_.sk_params["batch_size"])
		"""

		# Define callbacks
		generator_callbacks = [
			ModelCheckpoint(outputs_path + "/models/discriminator_model_checkpoint.h5", monitor='val_acc',
							verbose=1, save_best_only=True, mode='auto'),
			EarlyStopping(monitor='acc', patience=5)]

		# update discriminator model weights
		d_history = d_model.fit(X, y, validation_split=cfg["Global"]["Data"]["val_size"],
			batch_size=cfg["Discriminator"]["Training"]["batch_size"],
			epochs=cfg["Discriminator"]["Training"]["n_epochs"], verbose=2,
			callbacks=generator_callbacks)

		#Define callbacks
		discriminator_callbacks = [ModelCheckpoint(outputs_path + "/models/generator_model_checkpoint.h5", monitor='val_acc',
						 verbose=1, save_best_only=True, mode='auto'),
		 EarlyStopping(monitor='acc', patience=5)]

		# update the generator via the discriminator's error
		g_history = gan_model.fit(X_gan, y_gan, validation_split=cfg["Global"]["Data"]["val_size"],
			  batch_size=cfg["Generator"]["Training"]["batch_size"],
			  epochs=cfg["Generator"]["Training"]["n_epochs"], verbose=2,
			  callbacks=discriminator_callbacks)
	return d_history, g_history

def train_r_model(outputs_path, r_model, train_input_data, train_output_data):

	# Define callbacks
	regressor_callbacks = [
		ModelCheckpoint(outputs_path + "/models/regressor_model_checkpoint.h5", monitor='val_acc',
						verbose=1, save_best_only=True, mode='auto'),
		EarlyStopping(monitor='acc', patience=5)]

	r_history = r_model.fit(train_input_data, train_output_data, validation_split=cfg["Global"]["Data"]["val_size"],
			batch_size=cfg["Regressor"]["Training"]["batch_size"],
			epochs=cfg["Regressor"]["Training"]["n_epochs"], verbose=2,
			callbacks=regressor_callbacks)

	return r_history

def r_test(r_model, r_test_input_data, r_test_output_data):

	r_history = r_model.evaluate(r_test_input_data, r_test_output_data, verbose = 2)

	return r_history