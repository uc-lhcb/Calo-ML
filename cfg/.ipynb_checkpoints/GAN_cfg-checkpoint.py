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
path_to_calo_ml = str(pathlib.Path().absolute())

cfg_global = {

    "Data": {
      "_df_path_comment": "Path to get the Calo data.",
      "df_path": path_to_calo_ml + "/data/inputs/CaloGan_photons.h5",
      "_test_size_comment": "% of the whole dataset to create de testing dataset.",
      "test_size": 0.15,
      "val_size_comment": "% of the whole dataset to create de testing dataset.",
      "val_size": 0.15,
      "_random_state_comment": "Seed to split dataset",
      "random_state": 0,
      "_model_id_path_comment": "Model identifier to have a model version control.",
      "model_id_path": path_to_calo_ml + "/cfg/model_version.txt",
      "_outputs_path_comment": "Path to store the outputs generated.",
      "outputs_path": path_to_calo_ml + "/data/outputs/"
    },

    "Training": {
      "_type_comment": "Type of training. Either 'exploration' or 'creation'. Epxploration is to explore the error"
                       " space iterating over many possibilities in the different hyperparameters (using searchgrid) and"
                       " Creation is to create a GAN when the error space already explored",
      #"type": "creation",
      "type": "exploration",
      "_models_to_create_comment" : "Number of models to train. Each model means an iteration over the config file, "
                                    "so to tune hyperparameters it is recommended to set at 1.",
      "models_to_create" : 1,
      "_reproducible_results_comment" : "Not working",
      "reproducible_results" : False,
      "_training_iterations_comment" : "Iterations between generator & discriminator. Both the training and testing datasets are splited according to this number.",
      "training_iterations" : 1,
      "_name_comment" : "Use existing models to retrain them",
      "discriminator_name" : "",
      "generator_name" : "",
      "regressor_name" : ""
    },

    "Architecture": {

    }
  }

cfg_generator = {

    "Data": {
      "_input_parameters_comment" : "Known values that can be added to latent space.",
      "input_parameters" : ["ParticlePoint", "ParticleMomentum"],
      "_add_input_parameters_comment" : "Add the known values to the random latent space.",
      "add_input_parameters" : False,
      "_input_length_comment" : "Length of latent space.",
      "input_length" : 500,
      "_noise_dist_comment" : "Random distribution to create the latent space. Currently only works with normalized_gaussian (mu = 0, sd = 1)",
      "noise_dist" : "normalized_gaussian"
    },

    "Training": {
      "_epochs_comment" : "Epochs to train the generator.",
      "epochs" : 1,
      "_epochs_grid_comment" : "List of the values to optimize the generator epochs.",
      "epochs_grid" : [5, 10],
      "_batch_size_comment" : "Batch size to train the generator.",
      "batch_size" : 256,
      "_batch_size_grid_comment" : "List of the values to optimize the generator batch size.",
      "batch_size_grid" : [128, 256],
      "_lr_comment" : "Learning rate to train the generator.",
      "lr" : 0.0002,
      "_lr_grid_comment" : "List of the values to optimize the generator learning rate.",
      "lr_grid" : [0.0002, 0.002],
      "_beta_1_comment" : "Beta 1 to train the generator.",
      "beta_1" : 0.5,
      #"_beta_1_grid_comment" : "List of the values to optimize the generator beta 1.",
      #"beta_1_grid" : [0.1, 0.5],
      "_loss_func_comment" : "Loss function to train the generator.",
      "loss_func" : "binary_crossentropy",
      #"_loss_func_grid_comment" : "List of the values to optimize the generator loss function.",
      #"loss_func_grid" : ["binary_crossentropy"],
      "_WGAN_comment" : "Use the WGAN loss function",
      "WGAN" : False
    },

    "Architecture": {

    }
  }

cfg_discriminator = {

    "Data": {
      "_input_params_comment" : "Known values of the energy deposited in the cells.",
      "input_params" : ["EnergyDeposit"],
      "_energy_deposit_input_shape_comment" : "Size of the known values of the energy deposited in the cells. WARNING: it should be the same as the discrimnator in_shape",
      "energy_deposit_input_shape" : [30, 30]
    },

    "Training": {
      "_epochs_comment" : "Epochs to train the discriminator.",
      "epochs" : 1,
      "_epochs_grid_comment" : "List of the values to optimize the discriminator epochs.",
      "epochs_grid" : [5, 50],
      "_batch_size_comment" : "Batch size to train the discriminator.",
      "batch_size" : 256,
      "_batch_size_grid_comment" : "List of the values to optimize the discriminator batch size.",
      "batch_size_grid" : [128, 256],
      "_lr_comment" : "Learning rate to train the discriminator.",
      "lr" : 0.0002,
      "_lr_grid_comment" : "List of the values to optimize the discriminator learning rate.",
      "lr_grid" : [0.0002, 0.002],
      "_beta_1_comment" : "Beta 1 to train the discriminator.",
      "beta_1" : 0.5,
      #"_beta_1_grid_comment" : "List of the values to optimize the discriminator beta 1.",
      #"beta_1_grid" : [0.1, 0.5],
      "_loss_func_comment" : "Loss function to train the discriminator.",
      "loss_func" : "binary_crossentropy",
      #"_loss_func_grid_comment" : "List of the values to optimize the discriminator loss function.",
      #"loss_func_grid" : ["binary_crossentropy"],
      "_metrics_comment" : "Metric to optimize the discriminator.",
      "metrics" : ["accuracy"],
      "_WGAN_comment" : "Use the WGAN loss function",
      "WGAN" : False
    },

    "Architecture": {
      "_in_shape_comment" : "Input shape of the discrimnator. WARNING: it should be the same as the discrimnator energy_deposit_input_shape.",
      "in_shape" : (30, 30, 1),
      "_parallel_pooling_comment" : "Add parallel pooling to the architecture. Currently not working.",
      "parallel_pooling" : False
    }
  }

cfg_regressor = {

    "Data": {
      "_output_parameters_comment" : "Values that regressor should reconstruct. Currently not working.",
      "output_parameters" : ["ParticlePoint", "ParticleMomentum"]
    },

    "Training": {
      "_epochs_comment" : "Epochs to train the regressor.",
      "epochs" : 1,
      "_epochs_grid_comment" : "List of the values to optimize the regressor epochs.",
      "epochs_grid" : [5, 50],
      "_batch_size_comment" : "Batch size to train the regressor.",
      "batch_size" : 256,
      "_batch_size_grid_comment" : "List of the values to optimize the regressor batch size.",
      "batch_size_grid" : [128, 256],
      "_lr_comment" : "Learning rate to train the regressor.",
      "lr" : 0.0002,
      "_lr_grid_comment" : "List of the values to optimize the regressor learning rate.",
      "lr_grid" : [0.0002, 0.002],
      "_beta_1_comment" : "Beta 1 to train the regressor.",
      "beta_1" : 0.5,
      #"_beta_1_grid_comment" : "List of the values to optimize the regressor beta 1.",
      #"beta_1_grid" : [0.1, 0.5],
      "_loss_func_comment" : "Loss function to train the regressor.",
      "loss_func" : "binary_crossentropy",
      #"_loss_func_grid_comment" : "List of the values to optimize the regressor loss function.",
      #"loss_func_grid" : ["binary_crossentropy"],
      "_metrics_comment" : "Metric to optimize the regressor.",
      "metrics" : ["accuracy"],
      "_WGAN_comment": "Use the WGAN loss function",
      "WGAN": False,
    },

    "Architecture": {
      "_regressor_comment" : "Add a regressor to the architecture which will be trained before the GAN",
      "regressor" : False,
      "_in_shape_comment" : "Input shape of the regressor.",
      "in_shape" : (30, 30, 1),
    }
  }

cfg = {
  "Global": cfg_global,

  "Generator": cfg_generator,

  "Discriminator": cfg_discriminator,

  "Regressor": cfg_regressor

}
