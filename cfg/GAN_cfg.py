import pathlib

class GAN_config():

    def __init__(self):
        self.path_to_calo_ml = str(pathlib.Path().absolute())

        self.models_to_create = 1
        self.create_resproducible_result = True
        # Use existing models to retrain them
        #discriminator_name = "0_discriminator_fold"
        self.discriminator_name = ""
        #generator_name = "0_generator_fold"
        self.generator_name = ""
        #regressor_name = "0_regressor_fold"
        self.regressor_name = ""

        # WGAN implica usar la funció de loss W_loss, per fer mateixa red que Yandex.
        self.WGAN = False
        self.regressor = False

        ########
        # DATA #
        ########
        self.df_path = self.path_to_calo_ml + '/data/CaloGan_photons.h5'

        # Generator input data
        # Add data to latent space
        self.g_input_parameters = ["ParticlePoint", "ParticleMomentum"]
        self.add_g_input_parameters = False
        self.g_input_length = 500 # length of latent space
        self.noise_dist = "normalized_gaussian" # mu = 0, sd = 1
        self.g_val_size = 0.15 # 15% of training

        # Discriminator input data
        self.d_input_params = ["EnergyDeposit"]
        self.energy_deposit_input_shape = [30,30]
        self.d_val_size = 0.15 # 15% of training

        # Regressor input data
        self.r_input_params = ["EnergyDeposit"]
        self.energy_deposit_input_shape = [30,30]
        self.r_val_size = 0.15 # 15% of training

        # Regressor output data
        self.r_output_parameters = ["ParticlePoint", "ParticleMomentum"]

        ############################
        # Training hyperparameters #
        ############################
        # Data
        self.training_iterations = 1
        self.model_id_path = self.path_to_calo_ml + "/cfg/model_version.txt"
        self.test_size = 0.15
        self.random_state = 0 # Seed to split dataset

        # Discriminator
        self.d_n_epochs = 1
        self.d_batch_size = 256
        self.d_lr = 0.0002
        self.d_beta_1 = 0.5
        self.d_loss_func = 'binary_crossentropy'
        self.d_metrics = ['accuracy']

        # Generator
        self.g_n_epochs = 2
        self.g_batch_size = 256
        self.g_lr = 0.0002
        self.g_beta_1 = 0.5
        self.g_loss_func = 'binary_crossentropy'

        # Regressor
        self.r_n_epochs = 2
        self.r_batch_size = 256
        self.r_lr = 0.0002
        self.r_beta_1 = 0.5
        self.r_loss_func = 'binary_crossentropy'
        self.r_metrics = ['accuracy']


        ################################
        # Architecture hyperparameters #
        ################################

        # Discriminator
        self.d_in_shape = (self.energy_deposit_input_shape[0], self.energy_deposit_input_shape[1], 1)
        self.parallel_pooling = True

        # Generator

        # Regressor
        self.r_in_shape = (self.energy_deposit_input_shape[0], self.energy_deposit_input_shape[1], 1)

        ###########
        # Outputs #
        ###########

        self.outputs_path = self.path_to_calo_ml + '/outputs/'

# def main():
#     return GAN_config()
#
# if __name__ == "__main__":
#     # execute only if run as a script
#     main()