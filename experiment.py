# for adding packages on windows - for linux i think it is just "export" instead of "set"
# set PYTHONPATH="$PYTHONPATH:/Calo-ML/helpers/"
# set PYTHONPATH="$PYTHONPATH:/Calo-ML/cfg/"

from helpers.training import *
from helpers.data import *
from models.GAN import *
from models.VAE import *
from cfg.GAN_cfg import *
from cfg.VAE_cfg import *

gan_params = GAN_config()
vae_params = VAE_config()

# Set seed for reproducible results
# Not working
if gan_params.create_resproducible_result:
	from numpy.random import seed
	seed(1)
	import tensorflow
	tensorflow.random.set_seed(2)


def main():

	experiment = "VAE"
	# experiment = "GAN"

	if experiment == "GAN":

		for model_number in range(gan_params.models_to_create):

			r_model, d_model, g_model, gan_model = create_models()

			#Read model_id & updated it
			with open(gan_params.model_id_path, "r") as f:
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

			with open(gan_params.model_id_path, "w") as f:
				f.seek(0)
				f.truncate()
				f.write(str(model_id + 1))
				f.close()

	elif experiment == "VAE":

		model_id = 0

		#Load data
		# load image data
		train_loader, val_loader = get_train_val_loaders()

		# itll work in most cases
		device = torch.device("cuda:0")

		# args are "n", number of channels per layer, and "z_dim", number of dimensions in the latent space. z_dim gets sandwiched between linear layers so it can be basically
		# whatever, kl term in loss only looks at this layer... i think?
		ಠ_ಠ = torch_VAE(vae_params.num_channels, vae_params.latent_dim).to(device)
		optimizer = optim.Adam(ಠ_ಠ.parameters(), lr=vae_params.lr)
		for epoch in range(vae_params.no_epochs):
			result = train_VAE(epoch, train_loader, ಠ_ಠ.to(device), optimizer, device)
			for out in result:
				save_to_mlflow(out, None)

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
