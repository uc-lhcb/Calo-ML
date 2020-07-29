import numpy as np
import pathlib
path_to_calo_ml = str(pathlib.Path().absolute())

models_to_create = 1
create_resproducible_result = False
# Use existing models to retrain them
#discriminator_name = "0_discriminator_fold"
discriminator_name = ""
#generator_name = "0_generator_fold"
generator_name = ""
#regressor_name = "0_regressor_fold"
regressor_name = ""

# WGAN implica usar la funció de loss W_loss, per fer mateixa red que Yandex.
WGAN = False
regressor = False

########
# DATA #
########
df_path = path_to_calo_ml + '/data/CaloGan_photons.h5'

# Generator input data
# Add data to latent space
g_input_parameters = ["ParticlePoint", "ParticleMomentum"]
add_g_input_parameters = False
g_input_length = 500 # length of latent space
noise_dist = "normalized_gaussian" # mu = 0, sd = 1
g_val_size = 0.15 # 15% of training

# Discriminator input data
d_input_params = ["EnergyDeposit"]
energy_deposit_input_shape = [30,30]
d_val_size = 0.15 # 15% of training

# Regressor input data
r_input_params = ["EnergyDeposit"]
energy_deposit_input_shape = [30,30]
r_val_size = 0.15 # 15% of training

# Regressor output data
r_output_parameters = ["ParticlePoint", "ParticleMomentum"]

############################
# Training hyperparameters #
############################
# Data
training_iterations = 1
model_id_path = path_to_calo_ml + "/cfg/model_version.txt"
test_size = 0.15
random_state = 0 # Seed to split dataset

# Discriminator
d_n_epochs = 1
d_batch_size = 256
d_batch_size_grid = np.array([1, 128, 256, 512, 1024])
d_batch_size_grid = [128, 256]
d_lr = 0.0002
d_beta_1 = 0.5
d_loss_func = 'binary_crossentropy'
d_metrics = ['accuracy']

# Generator
g_n_epochs = 1
g_batch_size = 256
g_lr = 0.0002
g_beta_1 = 0.5
g_loss_func = 'binary_crossentropy'

# Regressor
r_n_epochs = 2
r_batch_size = 256
r_lr = 0.0002
r_beta_1 = 0.5
r_loss_func = 'binary_crossentropy'
r_metrics = ['accuracy']


################################
# Architecture hyperparameters #
################################

# Discriminator
d_in_shape = (energy_deposit_input_shape[0], energy_deposit_input_shape[1], 1)
parallel_pooling = True

# Generator

# Regressor
r_in_shape = (energy_deposit_input_shape[0], energy_deposit_input_shape[1], 1)

###########
# Outputs #
###########

outputs_path = path_to_calo_ml + '/outputs/'
