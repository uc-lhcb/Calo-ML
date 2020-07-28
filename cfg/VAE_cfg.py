import pathlib

class VAE_config():

    def __init__(self):
        self.path_to_calo_ml = str(pathlib.Path().absolute())

        # Data & model configuration
        self.img_width = 30
        self.img_height = 30
        self.batch_size = 128
        self.no_epochs = 10
        self.validation_split = 0.2
        self.verbosity = 1
        self.latent_dim = 100
        self.num_channels = 32
        self.input_shape = (self.img_height, self.img_width, self.num_channels)

        self.models_to_create = 1
        self.create_resproducible_result = True
        # discriminator_name = "0_discriminator_fold"
        self.discriminator_name = ""
        # generator_name = "0_generator_fold"
        self.generator_name = ""
        ########
        # DATA #
        ########
        self.df_path = self.path_to_calo_ml + '/data/CaloGan_photons.h5'

        # Generator input data
        self.g_input_parameters = ["ParticlePoint", "ParticleMomentum"]
        self.add_g_input_parameters = False
        self.g_input_length = 500 # length of training example
        self.noise_dist = "normalized_gaussian" # mu = 0, sd = 1
        self.g_val_size = 0.15 # 15% of training

        # Discriminator input data
        self.d_input_params = ["EnergyDeposit"]
        self.energy_deposit_input_shape = [30, 30]
        self.d_val_size = 0.15 # 15% of training

        ############################
        # Training hyperparameters #
        ############################
        # Data
        self.training_iterations = 20
        self.model_id_path = self.path_to_calo_ml + "/cfg/model_version.txt"
        self.test_size = 0.15
        self.random_state = 0 # Seed to split dataset

        # Note(will) what does this do?
        # Discriminator
        self.d_n_epochs = 2
        self.d_batch_size = 256
        self.d_lr = 0.0002
        self.d_beta_1 = 0.5
        self.d_loss_func = 'binary_crossentropy'
        self.d_metrics = ['accuracy']

        # Generator
        self.g_n_epochs = 5
        self.g_batch_size = 256
        self.g_lr = 0.0002
        self.g_beta_1 = 0.5
        self.g_loss_func = 'binary_crossentropy'

        # VAE
        self.lr = 5e-4

        ################################
        # Architecture hyperparameters #
        ################################

        # Discriminator
        self.d_in_shape = (self.energy_deposit_input_shape[0], self.energy_deposit_input_shape[1], 1)

        # Generator


        ###########
        # Outputs #
        ###########

        self.outputs_path = self.path_to_calo_ml + '/outputs/'


# def main():
#     return VAE_config()
#
# if __name__ == "__main__":
#     # execute only if run as a script
#     main()