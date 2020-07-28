import matplotlib.pyplot as plt
# from cfg.GAN_cfg import *
# from cfg.VAE_cfg import *
import pandas as pd
from os import path
import numpy as np


# Function to plot rowsXcolumns images from startiing point rand.
def plot_energy_grids(energy_grids_to_plot, rows=2, columns=2, name=""):
    fig=plt.figure(figsize=(12, 12))
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(energy_grids_to_plot[i].reshape((energy_deposit_input_shape[0], energy_deposit_input_shape[1])))
    plt.savefig(outputs_path + "images/" + name)
    #plt.show()


def save_model(model, name = "model"):
    # serialize model to JSON
    model_json = model.to_json()
    with open(outputs_path + "models/" + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(outputs_path + "models/" + name + ".h5")
    print("Saved model " + str(name) + " to disk")


def save_experiment_conf(name, model_id, g_train_history, d_train_history, g_test_history, d_test_history):

    if not path.exists(outputs_path + "cfg_data.xlsx"):

        generator_params = ['generator_n_epochs', 'generator_batch_size', 'generator_lr', 'generator_beta_1',
                             'generator_loss_func', 'generator test loss', 'generator test accuracy']
        dicscriminator_params = ['discriminator_n_epochs', 'discriminator_batch_size', 'discriminator_lr',
                                  'discriminator_beta_1', 'discriminator_loss_func', 'discriminator_metrics',
                                 'discriminator test loss', 'discriminator test accuracy']
        dataset_params = ['model_id', 'test_size', 'random_state']
        cfg_data = pd.DataFrame(columns = ['Name'] + dataset_params + generator_params + dicscriminator_params)

    else:
        cfg_data = pd.read_excel(outputs_path + "cfg_data.xlsx")
        cfg_data = cfg_data.drop(['Unnamed: 0'], axis=1)

    data = [name, model_id, test_size, random_state, g_n_epochs, g_batch_size, g_lr,
            g_beta_1, g_loss_func, g_test_history[0], g_test_history[1], d_n_epochs, d_batch_size, d_lr,
            d_beta_1, d_loss_func, d_metrics, d_test_history[0], d_test_history[1]]
    cfg_data.loc[name] = data
    cfg_data.to_excel(outputs_path + "cfg_data.xlsx")

def generate_metrics(gan_model, g_model, d_model, model_id, g_train_history, d_train_history, g_test_history, d_test_history):

    model_name = str(model_id) + "_gan_fold"
    save_model(d_model, name=str(model_id) + "_discriminator_fold")
    save_model(g_model, name=str(model_id) + "_generator_fold")
    save_model(gan_model, name=model_name)
    save_experiment_conf(model_name, model_id, g_train_history, d_train_history, g_test_history, d_test_history)


def myprint(s):
    with open('./outputs/modelsummary.txt', 'w+') as f:
        print(s, file=f)



# =================
# Results visualization
# Credits for original visualization code: https://keras.io/examples/variational_autoencoder_deconv/
# (François Chollet).
# Adapted to accomodate this VAE.
# =================
def viz_latent_space(encoder, data):
    input_data, target_data = data
    mu, _, _ = encoder.predict(input_data)
    plt.figure(figsize=(8, 10))
    plt.scatter(mu[:, 0], mu[:, 1], c=target_data)
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    plt.colorbar()
    plt.show()

def viz_decoded(encoder, decoder, data):
    num_samples = 15
    figure = np.zeros((img_width * num_samples, img_height * num_samples, num_channels))
    grid_x = np.linspace(-4, 4, num_samples)
    grid_y = np.linspace(-4, 4, num_samples)[::-1]
    for i, yi in enumerate(grid_y):
      for j, xi in enumerate(grid_x):
          z_sample = np.array([[xi, yi]])
          x_decoded = decoder.predict(z_sample)
          digit = x_decoded[0].reshape(img_width, img_height, num_channels)
          figure[i * img_width: (i + 1) * img_width,
                  j * img_height: (j + 1) * img_height] = digit
    plt.figure(figsize=(10, 10))
    start_range = img_width // 2
    end_range = num_samples * img_width + start_range + 1
    pixel_range = np.arange(start_range, end_range, img_width)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel('z - dim 1')
    plt.ylabel('z - dim 2')
    # matplotlib.pyplot.imshow() needs a 2D array, or a 3D array with the third dimension being of shape 3 or 4!
    # So reshape if necessary
    fig_shape = np.shape(figure)
    if fig_shape[2] == 1:
        figure = figure.reshape((fig_shape[0], fig_shape[1]))
    # Show image
    plt.imshow(figure)
    plt.show()

