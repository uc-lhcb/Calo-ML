from helpers import training
from helpers import data
from helpers import results
from models import GAN
from cfg.GAN_cfg import cfg


# Set seed for reproducible results
# Not working
if cfg["Global"]["Training"]["reproducible_results"]:
	from numpy.random import seed
	seed(1)
	import tensorflow
	tensorflow.random.set_seed(2)


def main():

	experiment = "GAN"

	if experiment == "GAN":

		for model_number in range(cfg["Global"]["Training"]["models_to_create"]):

			model_id, outputs_path = data.set_up_data(model_number)

			r_model, d_model, g_model, gan_model = GAN.create_models(outputs_path)

			# Number to identify the model
			print("model_id", model_id)


			# load image data
			g_train_input_data, g_test_input_data, d_train_input_data, d_test_input_data = data.load_real_samples(
				outputs_path)
			# train model
			d_train_history, g_train_history = training.train(outputs_path, g_model, d_model, gan_model, g_train_input_data, d_train_input_data, model_id)
			# test model
			g_test_history, d_test_history = training.test(outputs_path, g_model, d_model, gan_model, g_test_input_data, d_test_input_data, model_id)
			# Save models & metrics
			results.generate_metrics(outputs_path, gan_model, g_model, d_model, model_id, g_train_history, d_train_history, g_test_history, d_test_history)

			with open(cfg["Global"]["Data"]["model_id_path"], "w") as f:
				f.seek(0)
				f.truncate()
				f.write(str(model_id + 1))
				f.close()

if __name__ == '__main__':
	main()
