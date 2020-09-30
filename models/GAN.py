# example of training the discriminator model on real and random mnist images
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.models import model_from_json
from keras.layers import ReLU
from keras.layers import BatchNormalization
from keras.layers import Cropping2D
from keras.layers import Dense
import keras
from cfg.GAN_cfg import cfg
from helpers.functions import *
from contextlib import redirect_stdout

print("GAN cfg: ", cfg.keys)


# define the standalone discriminator model
def define_discriminator(outputs_path):
	model = Sequential()
	model.add(Conv2D(32, (4, 4), strides=(1, 1), padding='same', input_shape=cfg["Discriminator"]["Architecture"]["in_shape"],
					 kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same',
					 kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(128, (4, 4), strides=(2, 2), padding='same',
					 kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Conv2D(256, (4, 4), strides=(2, 2), padding='same',
					 kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros'))
	model.add(LeakyReLU(alpha=0.2))

	model.add(Flatten())

	model.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=1337),
					bias_initializer='zeros'))
	# compile model
	opt = Adam(lr=cfg["Discriminator"]["Training"]["lr"], beta_1=cfg["Discriminator"]["Training"]["beta_1"])
	model.compile(loss=cfg["Discriminator"]["Training"]["loss_func"], optimizer=opt, metrics=cfg["Discriminator"]["Training"]["metrics"])
	with open(outputs_path+ 'd_modelsummary.txt', 'w') as f:
		with redirect_stdout(f):
			model.summary()

	return model


# define the standalone discriminator model
def define_regressor(outputs_path):

	inputs = keras.Input(shape=(30, 30, 1))

	C2D_1 = Conv2D(32, (4, 4), strides=(1, 1), padding='same', input_shape=cfg["Regressor"]["Architecture"]["in_shape"],
					 kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros')(inputs)
	LR_1 = LeakyReLU(alpha=0.2)(C2D_1)

	C2D_2 = Conv2D(64, (4, 4), strides=(2, 2), padding='same',
					 kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros')(LR_1)
	LR_2 = LeakyReLU(alpha=0.2)(C2D_2)

	C2D_3 = Conv2D(128, (4, 4), strides=(2, 2), padding='same',
					 kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros')(LR_2)
	LR_3 = LeakyReLU(alpha=0.2)(C2D_3)

	C2D_4 = Conv2D(256, (4, 4), strides=(2, 2), padding='same',
					 kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros')(LR_3)
	LR_4 = LeakyReLU(alpha=0.2)(C2D_4)

	flatten = Flatten()(LR_4)

	output_1 = Dense(1, kernel_initializer=keras.initializers.RandomNormal(seed=1337),
					bias_initializer='zeros')(flatten)
	output_2 = Dense(1, kernel_initializer=keras.initializers.RandomNormal(seed=1337),
					bias_initializer='zeros')(flatten)
	output_3 = Dense(1, kernel_initializer=keras.initializers.RandomNormal(seed=1337),
					bias_initializer='zeros')(flatten)
	output_4 = Dense(1, kernel_initializer=keras.initializers.RandomNormal(seed=1337),
					bias_initializer='zeros')(flatten)
	output_5 = Dense(1, kernel_initializer=keras.initializers.RandomNormal(seed=1337),
					bias_initializer='zeros')(flatten)

	model = keras.Model(inputs=inputs, outputs=[output_1, output_2, output_3, output_4, output_5])

	# compile model
	opt = Adam(lr=cfg["Regressor"]["Training"]["lr"], beta_1=cfg["Regressor"]["Training"]["beta_1"])
	model.compile(loss=cfg["Regressor"]["Training"]["loss_func"], optimizer=opt, metrics=cfg["Regressor"]["Training"]["metrics"])

	with open(outputs_path + 'r_modelsummary.txt', 'w') as f:
		with redirect_stdout(f):
			model.summary()
	return model


# define the standalone generator model
def define_generator(outputs_path):

	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=cfg["Generator"]["Data"]["input_length"], kernel_initializer=keras.initializers.RandomNormal(seed=1337),
    bias_initializer='zeros'))
	model.add(Reshape((4, 4, 256)))

	model.add(UpSampling2D())
	model.add(Conv2D(128, (4,4), strides=(1,1), padding='same',
							  kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros'))
	model.add(BatchNormalization())
	model.add(ReLU())

	model.add(UpSampling2D())
	model.add(Conv2D(64, (4, 4), strides=(1, 1), padding='same',
					 kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros'))
	model.add(BatchNormalization())
	model.add(ReLU())

	model.add(UpSampling2D())
	model.add(Conv2D(32, (4, 4), strides=(1, 1), padding='same',
					 kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros'))
	model.add(BatchNormalization())
	model.add(ReLU())

	model.add(Cropping2D([1,1]))

	model.add(Conv2D(1, (4, 4), strides=(1, 1), padding='same',
					 kernel_initializer=keras.initializers.RandomNormal(seed=1337), bias_initializer='zeros'))
	model.add(ReLU())

	with open(outputs_path + 'g_modelsummary.txt', 'w') as f:
		with redirect_stdout(f):
			model.summary()

	return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=cfg["Generator"]["Training"]["lr"], beta_1=cfg["Generator"]["Training"]["beta_1"])
	if cfg["Generator"]["Training"]["WGAN"]:
		model.compile(loss=wasserstein_loss, optimizer=opt, metrics=cfg["Discriminator"]["Training"]["metrics"])
	else:
		model.compile(loss=cfg["Generator"]["Training"]["loss_func"], optimizer=opt, metrics=cfg["Discriminator"]["Training"]["metrics"])
	model.summary()
	return model


def load_model(outputs_path, model_to_load):
	# load json and create model
	path = outputs_path + "models/" + model_to_load
	json_file = open(path + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(path + ".h5")

	if "discriminator" in model_to_load:
		opt = Adam(lr=cfg["Discriminator"]["Training"]["lr"], beta_1=cfg["Discriminator"]["Training"]["beta_1"])
		loaded_model.compile(loss=cfg["Discriminator"]["Training"]["loss_func"], optimizer=opt,
							 metrics=cfg["Discriminator"]["Training"]["metrics"])

	return loaded_model

def create_models(outputs_path):

	# create the discriminator
	if cfg["Global"]["Training"]["discriminator_name"] != "":
		d_model = load_model(outputs_path, cfg["Global"]["Training"]["discriminator_name"])
	else:
		d_model = define_discriminator(outputs_path)

	# create the generator
	if cfg["Global"]["Training"]["generator_name"] != "":
		g_model = load_model(outputs_path, cfg["Global"]["Training"]["generator_name"])
	else:
		g_model = define_generator(outputs_path)

	# create the generator
	if cfg["Global"]["Training"]["regressor_name"] != "":
		g_model = load_model(outputs_path, cfg["Global"]["Training"]["regressor_name"])
		g_model = load_model(outputs_path, cfg["Global"]["Training"]["regressor_name"])
	else:
		r_model = define_regressor(outputs_path)

	# create the gan
	gan_model = define_gan(g_model, d_model)

	return r_model, d_model, g_model, gan_model