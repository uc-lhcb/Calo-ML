import mlflow
import torch
import numpy as np
import math

def start_mlflow_experiment(experiment_name, model_store):
    '''
    model_store options: pv-finder, lane-finder, Calo-ML (not yet tho)
    '''
    if model_store == 'Calo-ML':
        raise NotImplementedError

    mlflow.set_experiment(experiment_name)


def save_to_mlflow(stats_dict: dict, step):
    '''
    Requires that the dictionary be structured as:
    Parameters have the previx "Param: ", metrics have "Metric: ", and artifacts have "Artifact: "
    It will ignore these tags.

    Example: {'Param: Parameters':106125, 'Metric: Training Loss':10.523}
    '''
    for key, value in stats_dict.items():
        if 'Param: ' in key:
            mlflow.log_param(key[7:], value)
        if 'Metric: ' in key:
            mlflow.log_metric(key[8:], value, step)
        if 'Artifact' in key:
            mlflow.log_artifact(value)
    # for key, value in vars(args).items():
    #     mlflow.log_param(key, value)


def count_parameters(model):
    """
    Counts the total number of parameters in a model
    Args:
        model (Module): Pytorch model, the total number of parameters for this model will be counted.

    Returns: Int, number of parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_full_state(model_to_update, optimizer_to_update, Path, freeze_weights=False):
    """
    Load the model and optimizer state_dict, and the total number of epochs
    The use case for this is if we care about the optimizer state_dict, which we do if we have multiple training
    sessions with momentum and/or learning rate decay. this will track the decay/momentum.

    Args:
            model_to_update (Module): Pytorch model with randomly initialized weights. These weights will be updated.
            optimizer_to_update (Module): Optimizer with your learning rate set.
            THIS FUNCTION WILL NOT UPDATE THE LEARNING RATE YOU SPECIFY.
            Path (string): If we are not training from scratch, this path should be the path to the "run_stats" file in the artifacts
            directory of whatever run you are using as a baseline.
            You can find the path in the MLFlow UI. It should end in /artifacts/run_stats


    Returns:
            Nothing

    Note:
            The model and optimizer will not be returned, rather the optimizer and module you pass to this function will be modified.
    """
    checkpoint = torch.load(Path)

    # freeze weights of the first model
    update_dict = {k: v for k, v in checkpoint['model'].items() if k in model_to_update.state_dict()}
    # do this so it does not use the learning rate from the previous run. this is unwanted behavior
    # in our scenario since we are not using a learning rate scheduler, rather we want to tune the learning
    # rate further after we have gotten past the stalling
    #     checkpoint['optimizer']['param_groups'][0]['lr'] = optimizer_to_update.state_dict()['param_groups'][0]['lr']
    #     optimizer_to_update.load_state_dict(checkpoint['optimizer'])

    # to go back to old behavior, just do checkpoint['model'] instead of update_dict
    model_to_update.load_state_dict(update_dict, strict=False)

    ct = 0
    if freeze_weights:
        for k, v in model_to_update.named_children():
            if ((k + '.weight' in checkpoint['model'].keys()) | (k + '.bias' in checkpoint['model'].keys())) & (
                    k != 'Dropout'):
                v.weight.requires_grad = False
                v.bias.requires_grad = False
                ct += 1

    print('we also froze {} weights'.format(ct))

    print('Of the ' + str(
        len(model_to_update.state_dict()) / 2) + ' parameter layers to update in the current model, ' + str(
        len(update_dict) / 2) + ' were loaded')


import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def transform(x):
    if isinstance(x, torch.Tensor):
        return torch.log(x.clone()+1)
    else:
        return np.log(x+1)

def inv_transform(x):
    if isinstance(x, torch.Tensor):
        return torch.exp(x.clone())-1
    else:
        return np.exp(x)-1

def reshape_energy_grid(energy_grid, new_shape=30):
    reshaped_grid = energy_grid.reshape((30, 30))
    return reshaped_grid

def create_cluster(energy_grid, num_of_cells):
    rows_0 = len(energy_grid)
    columns_0 = len(energy_grid[0])
    #print("original sizes: ", rows_0, columns_0)
    cells_dim = int(rows_0/math.sqrt(num_of_cells))
    #print("cells_dim: ", cells_dim)
    columns_1 = int(columns_0/cells_dim)
    rows_1 = int(rows_0/cells_dim)
    
    new_grid = np.zeros((rows_1,columns_1))

    for i in range(rows_1):
        for j in range(columns_1):
            cluster = energy_grid[int(i*cells_dim):int((i+1)*cells_dim-1),int(j*cells_dim):int((j+1)*cells_dim-1)]
            new_grid[i][j] = sum(sum(cluster))/len(cluster) 
    
    return new_grid
        
# Function to plot rowsXcolumns images from startiing point rand.
def plot_energy_grid(energy_grid_to_plot, rows=2, columns=2):
    fig=plt.figure(figsize=(12, 12))
    for i in range(1, columns*rows +1):
        plt.imshow(energy_grid_to_plot)
    plt.show()

def plot_3d_bar(data, cells, title, path): 
    # setup the figure and axes
    fig = plt.figure(figsize=(32, 12))
    ax1 = fig.add_subplot(121, projection='3d')
    
    # get the coordinates
    x = list(range(0, cells))
    y = list(range(0, cells))
    _xx, _yy = np.meshgrid(x, y)
    x, y = _xx.ravel(), _yy.ravel()

    #Get top & down values
    top = data.ravel()
    bottom = np.zeros_like(top)
    width = depth = 1

    norm = colors.Normalize(top.min(), top.max())
    cls = cm.jet(norm(top))
    ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color=cls)
    ax1.set_title(title)

    fig.savefig(path)

import statistics 

def compute_avg_energy(reshapaed_energy_grids):
    total_energy = 0
    for counter, image in enumerate(reshapaed_energy_grids):
        total_energy += sum(sum(image))
    total_energy = total_energy/50000
    print("The average energy per image is: " + str(total_energy))
    return total_energy

def find_baricenter(image):
    baricenter = [0,0]
    total_energy = sum(sum(image))
    for row, energy_row in enumerate(image):
        for column, energy in enumerate(energy_row):
            if energy > 0.0:
                baricenter[0] += (row+1)*energy/total_energy
                baricenter[1] += (column+1)*energy/total_energy
    #print("Image energy: " + str(total_energy))
    #print("Image baricenter: " + str(baricenter))
    return baricenter, total_energy

def iterate_image(distances, image):
    baricenter, total_energy = find_baricenter(image)
    for row, energy_row in enumerate(image):
        for column, energy in enumerate(energy_row):
            if energy > 0.0:
                #print("The energy in the cell " + str(row) + "," + str(column) + " is " + str(energy))
                x = math.pow((row+1)*energy/total_energy - baricenter[0],2)*energy/total_energy
                y = math.pow((column+1)*energy/total_energy - baricenter[1],2)*energy/total_energy
                distances.append(math.sqrt(x + y))

def compute_avg_distance(reshapaed_energy_grids):
    avg_distances_per_images = []
    sd_distances_per_images = []
    for counter, image in enumerate(reshapaed_energy_grids):
        if counter%5000 == 0.0:
            print("Computing image number " + str(counter) + " of " + str(len(reshapaed_energy_grids)))
        distances = []
        for image_direction in [image, np.flip(image), np.transpose(image), np.flip(np.transpose(image))]:
            iterate_image(distances, image_direction)

        avg_distances_per_images.append(sum(distances)/len(distances))
        sd_distances_per_images.append(statistics.stdev(distances))

    avg_whole_dataset_distances = sum(avg_distances_per_images)/len(avg_distances_per_images)
    sd_distances_per_images_squared = []
    for sd in sd_distances_per_images:
        sd_distances_per_images_squared.append(math.pow(sd,2))
    avg_whole_dataset_sd = math.sqrt(sum(sd_distances_per_images_squared))/len(avg_distances_per_images)
    
    print("The avg distance is " + str(avg_whole_dataset_distances) + " cells with an error of " + str(avg_whole_dataset_sd) + " cells.")
    print("The avg distance is " + str(avg_whole_dataset_distances*2) + " cms with an error of " + str(avg_whole_dataset_sd*2) + " cms.")
