from helpers.results import *
from cfg.GAN_cfg import *
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from models.GAN import *


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
	vae_history = vae.fit(input_train, input_train, epochs=no_epochs, batch_size=batch_size, validation_split=validation_split)

	return vae_history

def train(g_model, d_model, gan_model, g_train_input_data, d_train_input_data, model_id):

	number_of_samples = d_train_input_data.shape[0]
	# get randomly selected 'real' samples
	x_real = d_train_input_data
	y_real = np.ones((number_of_samples, 1))

	for i in range(training_iterations):

		first_split_point = int((i)*number_of_samples/training_iterations)
		second_split_point = int((i+1)*number_of_samples/training_iterations)
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

		"""
		# Code to tune hyperparameters
		keras_estimator = KerasClassifier(build_fn=define_discriminator, verbose=1)
		param_grid = dict(batch_size=d_batch_size_grid)
		grid = GridSearchCV(estimator=keras_estimator, param_grid=param_grid)
		grid.fit(X, y)
		print(grid.best_score_)
		print(grid.best_estimator_.sk_params["batch_size"])
		"""

		# update discriminator model weights
		d_history = d_model.fit(X, y, validation_split=d_val_size, batch_size=d_batch_size, epochs=d_n_epochs, verbose=2)
		# update the generator via the discriminator's error
		g_history = gan_model.fit(X_gan, y_gan, validation_split=g_val_size, batch_size=g_batch_size, epochs=g_n_epochs, verbose=2)

	return d_history, g_history

def train_r_model(r_model, train_input_data, train_output_data):

	r_history = r_model.fit(train_input_data, train_output_data, validation_split=r_val_size, batch_size=r_batch_size, epochs=r_n_epochs,
							  verbose=2)

	return r_history

def r_test(r_model, r_test_input_data, r_test_output_data):

	r_history = r_model.evaluate(r_test_input_data, r_test_output_data, verbose = 2)

	return r_history