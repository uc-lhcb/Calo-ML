import pathlib
path_to_calo_ml = str(pathlib.Path().absolute())


# Data & model configuration
img_width = 30
img_height = 30
img_width, img_height = img_width, img_height
batch_size = 128
no_epochs = 10
validation_split = 0.2
verbosity = 1
latent_dim = 2
num_channels = 1
input_shape = (img_height, img_width, num_channels)


models_to_create = 1
create_resproducible_result = True
#discriminator_name = "0_discriminator_fold"
discriminator_name = ""
#generator_name = "0_generator_fold"
generator_name = ""
########
# DATA #
########
df_path = path_to_calo_ml + '/data/CaloGan_photons.h5'

# Generator input data
g_input_parameters = ["ParticlePoint", "ParticleMomentum"]
add_g_input_parameters = False
g_input_length = 500 # length of training example
noise_dist = "normalized_gaussian" # mu = 0, sd = 1
g_val_size = 0.15 # 15% of trianing

# Discriminator input data
d_input_params = ["EnergyDeposit"]
energy_deposit_input_shape = [30,30]
d_val_size = 0.15 # 15% of training

############################
# Training hyperparameters #
############################
# Data
training_iterations = 20
model_id_path = path_to_calo_ml + "/cfg/model_version.txt"
test_size = 0.15
random_state = 0 # Seed to split dataset

# Discriminator
d_n_epochs = 2
d_batch_size = 256
d_lr = 0.0002
d_beta_1 = 0.5
d_loss_func = 'binary_crossentropy'
d_metrics = ['accuracy']

# Generator
g_n_epochs = 5
g_batch_size = 256
g_lr = 0.0002
g_beta_1 = 0.5
g_loss_func = 'binary_crossentropy'


################################
# Architecture hyperparameters #
################################

# Discriminator
d_in_shape = (energy_deposit_input_shape[0], energy_deposit_input_shape[1], 1)

# Generator


###########
# Outputs #
###########

outputs_path = path_to_calo_ml + '/outputs/'
