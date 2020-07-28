import mlflow
import torch

def start_mlflow_experiment(experiment_name, model_store):
    '''
    model_store options: pv-finder, lane-finder, Calo-ML (not yet tho)
    '''
    if model_store == 'Calo-ML':
        raise NotImplementedError

    mlflow.set_experiment(experiment_name)


def save_to_mlflow(stats_dict: dict):
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
            mlflow.log_metric(key[8:], value)
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



