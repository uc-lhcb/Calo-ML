# Calo-ML
Calo ML collaboration project with team from Barcelona 

The main file you will use is train-VAE.ipynb. This just runs a single experiment and logs to MLFlow. If you are not will (likely), change the storage location for the MLFlow logs. (set_tracking_uri). Change it to whatever your group is using for other projects, with a new experiment name.


#### Viewdata.ipynb and EgorFedor-2.ipynb
These are notebooks from Dr. Ratnikov that he provided me with. They go over how he generates useful figures of merit

#### ViewData.ipynb
This was a notebook that Oriol made to look at the predicted vs truth events

#### simple_generative_models.ipynb
This was a simple experiment done to see how close simple generative models (sampling from familiar distributions) could get us with this task
Answer: Not very 

## Important files

#### helpers/data.py
This holds the dataloader

#### helpers/training.py
This holds the train_vae, where the entire training pipeline is contained. 

#### models/VAE.py
This has the model architecture. Between this, training.py, and data.py, and some helper functions, you have everything you need to integrate my training procedures into your own codebase.

## Useful Links
https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
https://developer.nvidia.com/tensorrt
https://arxiv.org/abs/1812.01319
https://docs.google.com/presentation/d/12MiKXNcPr1zMTLQwpjC_eg5kjPPXz4Yn04sTQuWUi_I/edit?usp=sharing