from helpers.training import *
from helpers.data import *
from models.GAN import *
from models.GAN import *
from cfg.GAN_cfg import *
from cfg.VAE_cfg import *

# Set seed for reproducible results
# Not working
if create_resproducible_result:
	from numpy.random import seed
	seed(1)
	import tensorflow
	tensorflow.random.set_seed(2)


def main():

	#experiment = "VAE"
	experiment = "GAN"

	if experiment == "GAN":

		for model_number in range(models_to_create):

			r_model, d_model, g_model, gan_model = create_models()

			#Read model_id & updated it
			with open(model_id_path, "r") as f:
				model_id = int(f.readlines()[0])
				f.close()
			# Number to identify the model
			print("model_id", model_id)


			# load image data
			g_train_input_data, g_test_input_data, d_train_input_data, d_test_input_data = load_real_samples(
				model_id)
			# train model
			d_train_history, g_train_history = train(g_model, d_model, gan_model, g_train_input_data, d_train_input_data, model_id)
			# test model
			g_test_history, d_test_history = test(g_model, d_model, gan_model, g_test_input_data, d_test_input_data, model_id)
			# Save models & metrics
			generate_metrics(gan_model, g_model, d_model, model_id, g_train_history, d_train_history, g_test_history, d_test_history)

			with open(model_id_path, "w") as f:
				f.seek(0)
				f.truncate()
				f.write(str(model_id + 1))
				f.close()

	elif experiment == "VAE":

		model_id = 0

		#Load data
		# load image data
		g_train_input_data, g_test_input_data, input_train, d_test_input_data = load_real_samples(model_id)
		# Load MNIST dataset
		"""
		(input_train, target_train), (input_test, target_test) = mnist.load_data()
		img_width, img_height = input_train.shape[1], input_train.shape[2]
		num_channels = 1
		input_train = input_train.reshape(input_train.shape[0], img_height, img_width, num_channels)
		input_train = input_train.astype('float32')
		input_train = input_train / 255
		

		# Create VAE model
		VAE = models.VAE.VAE("test")
		A_model = VAE.define_autoencoder()
		D_model = VAE.define_decoder()
		VAE_model = VAE.create_vae()

		#Train vae
		train_vae(VAE_model, input_train)

		# Plot results

		data = (input_test, target_test)
		viz_latent_space(encoder, data)
		viz_decoded(encoder, decoder, data)
		"""

if __name__ == '__main__':
	main()
