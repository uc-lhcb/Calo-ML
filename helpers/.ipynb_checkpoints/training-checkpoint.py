
import helpers.results
from cfg.GAN_cfg import *
from cfg.VAE_cfg import *
import numpy as np

# for will's stuff
import torch
from torch import nn, optim
from helpers.utilities import count_parameters, plot_3d_bar, save_to_mlflow
from helpers.utilities import transform, inv_transform
from torchvision import utils
from tqdm import tqdm
import torch.nn.functional as F

# gan_params = GAN_config()
vae_params = cfg

# From Oriol, no longer maintained
def test(g_model, d_model, gan_model, g_test_input_data, d_test_input_data, model_id):
    
    number_of_samples = d_test_input_data.shape[0]
    # get randomly selected 'real' samples
    x_real = d_test_input_data
    y_real = np.ones((number_of_samples, 1))
    # generate 'fake' examples
    # generate points in latent space
    X_fake = g_model.predict(g_test_input_data)

    y_fake = np.zeros((number_of_samples, 1))
    # create training set for the discriminator
    X, y = np.vstack((x_real, X_fake)), np.vstack((y_real, y_fake))
    # update discriminator model weights
    d_history = d_model.evaluate(X, y, verbose=2)
    print("discriminator history", d_history)
    # prepare points in latent space as input for the generator
    X_gan = g_test_input_data
    # create inverted labels for the fake samples
    y_gan = np.ones((number_of_samples, 1))
    # update the generator via the discriminator's error
    g_history = gan_model.evaluate(X_gan, y_gan, verbose=2)

    results.plot_energy_grids(X_fake, name=str(model_id) + "_generated_images")

    print("generator history", g_history)

    return d_history, g_history


# From Oriol, no longer maintained
def train_vae(vae, input_train):
    # Train autoencoder
    vae_history = vae.fit(input_train, input_train, epochs=vae_params.no_epochs, batch_size=vae_params.batch_size,
                          validation_split=vae_params.validation_split)

    return vae_history


# From Oriol, no longer maintained
def train(g_model, d_model, gan_model, g_train_input_data, d_train_input_data, model_id):
    number_of_samples = d_train_input_data.shape[0]
    # get randomly selected 'real' samples
    x_real = d_train_input_data
    y_real = np.ones((number_of_samples, 1))

    for i in range(gan_params.training_iterations):
        first_split_point = int((i) * number_of_samples / gan_params.training_iterations)
        second_split_point = int((i + 1) * number_of_samples / gan_params.training_iterations)
        # generate 'fake' examples
        # generate points in latent space
        # x_input = generate_latent_points(latent_dim, number_of_samples)
        X_fake = g_model.predict(g_train_input_data[first_split_point:second_split_point])
        y_fake = np.zeros((second_split_point - first_split_point, 1))
        results.plot_energy_grids(X_fake, name=str(model_id) + "_fake_images_" + str(i))

        X, y = np.vstack((x_real[first_split_point:second_split_point], X_fake)), \
               np.vstack((y_real[first_split_point:second_split_point], y_fake))

        # prepare points in latent space as input for the generator
        # X_gan = generate_latent_points(latent_dim, number_of_samples)
        X_gan = g_train_input_data[first_split_point:second_split_point]
        # create inverted labels for the fake samples
        y_gan = np.ones((second_split_point - first_split_point, 1))

        # update discriminator model weights
        d_history = d_model.fit(X, y, validation_split=gan_params.d_val_size, batch_size=gan_params.d_batch_size,
                                epochs=gan_params.d_n_epochs, verbose=2)
        # update the generator via the discriminator's error
        g_history = gan_model.fit(X_gan, y_gan, validation_split=gan_params.g_val_size,
                                  batch_size=gan_params.g_batch_size, epochs=gan_params.g_n_epochs, verbose=2)

    return d_history, g_history

# From Oriol, no longer maintained
def train_r_model(r_model, train_input_data, train_output_data):
    r_history = r_model.fit(train_input_data, train_output_data, validation_split=r_val_size, batch_size=r_batch_size,
                            epochs=r_n_epochs,
                            verbose=2)

    return r_history

from helpers.results import plot_energy_grids
from cfg.GAN_cfg import cfg
# from models import GAN
# from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import GridSearchCV
# from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

# From Oriol, no longer maintained
def test(outputs_path, g_model, d_model, gan_model, g_test_input_data, d_test_input_data, model_id):

	number_of_samples = d_test_input_data.shape[0]
	# get randomly selected 'real' samples
	x_real = d_test_input_data
	y_real = np.ones((number_of_samples, 1))
	# generate 'fake' examples
	# generate points in latent space
	X_fake = g_model.predict(g_test_input_data)

	y_fake = np.zeros((number_of_samples, 1))
	# create training set for the discriminator
	X, y = np.vstack((x_real, X_fake)), np.vstack((y_real, y_fake))
	# update discriminator model weights
	d_history = d_model.evaluate(X, y, verbose=2)
	print("discriminator history", d_history)
	# prepare points in latent space as input for the generator
	X_gan = g_test_input_data
	# create inverted labels for the fake samples
	y_gan = np.ones((number_of_samples, 1))
	# update the generator via the discriminator's error
	g_history = gan_model.evaluate(X_gan, y_gan, verbose = 2)

	plot_energy_grids(outputs_path, X_fake, name="generated_images")

	print("generator history", g_history)

	return d_history, g_history

# From Oriol, no longer maintained
def train(outputs_path, g_model, d_model, gan_model, g_train_input_data, d_train_input_data, model_id):

	number_of_samples = d_train_input_data.shape[0]
	# get randomly selected 'real' samples
	x_real = d_train_input_data
	y_real = np.ones((number_of_samples, 1))

	if cfg["Global"]["Training"]["type"] == "creation":

		for i in range(cfg["Global"]["Training"]["training_iterations"]):

			first_split_point = int((i)*number_of_samples/cfg["Global"]["Training"]["training_iterations"])
			second_split_point = int((i+1)*number_of_samples/cfg["Global"]["Training"]["training_iterations"])
			# generate 'fake' examples
			# generate points in latent space
			# x_input = generate_latent_points(latent_dim, number_of_samples)
			X_fake = g_model.predict(g_train_input_data[first_split_point:second_split_point])
			y_fake = np.zeros((second_split_point - first_split_point, 1))
			plot_energy_grids(outputs_path, X_fake, name="fake_images_" + str(i))

			X, y = np.vstack((x_real[first_split_point:second_split_point], X_fake)),\
				   np.vstack((y_real[first_split_point:second_split_point], y_fake))

			# prepare points in latent space as input for the generator
			# X_gan = generate_latent_points(latent_dim, number_of_samples)
			X_gan = g_train_input_data[first_split_point:second_split_point]
			# create inverted labels for the fake samples
			y_gan = np.ones((second_split_point - first_split_point, 1))

			# Define callbacks
			generator_callbacks = [
				ModelCheckpoint(outputs_path + "/models/discriminator_model_checkpoint.h5", monitor='val_acc',
								verbose=1, save_best_only=True, mode='auto'),
				EarlyStopping(monitor='acc', patience=5)]

			# update discriminator model weights
			d_history = d_model.fit(X, y, validation_split=cfg["Global"]["Data"]["val_size"],
									batch_size=cfg["Discriminator"]["Training"]["batch_size"],
									epochs=cfg["Discriminator"]["Training"]["epochs"], verbose=2,
									callbacks=generator_callbacks)

			# Define callbacks
			discriminator_callbacks = [
				ModelCheckpoint(outputs_path + "/models/generator_model_checkpoint.h5", monitor='val_acc',
								verbose=1, save_best_only=True, mode='auto'),
				EarlyStopping(monitor='acc', patience=5)]

			# update the generator via the discriminator's error
			g_history = gan_model.fit(X_gan, y_gan, validation_split=cfg["Global"]["Data"]["val_size"],
									  batch_size=cfg["Generator"]["Training"]["batch_size"],
									  epochs=cfg["Generator"]["Training"]["epochs"], verbose=2,
									  callbacks=discriminator_callbacks)

			return d_history, g_history

	elif cfg["Global"]["Training"]["type"] == "exploration":

		#Data
		X_fake = g_model.predict(g_train_input_data)
		y_fake = np.zeros((number_of_samples, 1))
		#plot_energy_grids(outputs_path, X_fake, name="fake_images")

		X, y = np.vstack((x_real, X_fake)), \
			   np.vstack((y_real, y_fake))

		X_gan = g_train_input_data
		# create inverted labels for the fake samples
		y_gan = np.ones((number_of_samples, 1))

		# Explore Discriminator
		print("Exploring the error space of the discriminator")
		keras_d_estimator = KerasClassifier(build_fn=GAN.define_discriminator, verbose=1)

		param_grid = dict()
		for train_param_key, train_param_value in cfg["Discriminator"]["Training"].items():
			if not "comment" in train_param_key and "grid" in train_param_key:
				param_grid[train_param_key.replace("_grid", "")] = train_param_value

		#param_grid = dict(batch_size=cfg["Discriminator"]["Training"]["batch_size_grid"])
		d_grid = GridSearchCV(estimator=keras_d_estimator, param_grid=param_grid, cv=3)
		d_grid.fit(X, y)
		print("Search space best score")
		print(d_grid.best_score_)
		print("Search space best parameters")
		print(d_grid.best_estimator_)
		#print(d_grid.best_estimator_.sk_params["batch_size"])

		# Explore Generator
		print("Exploring the error space of the generator")
		keras_g_estimator = KerasClassifier(build_fn=GAN.define_experimental_gan, verbose=1)
		param_grid = dict(batch_size=cfg["Generator"]["Training"]["batch_size_grid"])
		g_grid = GridSearchCV(estimator=keras_g_estimator, param_grid=param_grid, cv=2)
		g_grid.fit(X_gan, y_gan)
		print("Search space best score")
		print(d_grid.best_score_)
		print("Search space best parameters")
		print(d_grid.best_estimator_)
		# print(d_grid.best_estimator_.sk_params["batch_size"])

		return d_grid, g_grid


# From Oriol, no longer maintained    
def train_r_model(outputs_path, r_model, train_input_data, train_output_data):

	# Define callbacks
	regressor_callbacks = [
		ModelCheckpoint(outputs_path + "/models/regressor_model_checkpoint.h5", monitor='val_acc',
						verbose=1, save_best_only=True, mode='auto'),
		EarlyStopping(monitor='acc', patience=5)]

	r_history = r_model.fit(train_input_data, train_output_data, validation_split=cfg["Global"]["Data"]["val_size"],
			batch_size=cfg["Regressor"]["Training"]["batch_size"],
			epochs=cfg["Regressor"]["Training"]["epochs"], verbose=2,
			callbacks=regressor_callbacks)


# From Oriol, no longer maintained
def r_test(r_model, r_test_input_data, r_test_output_data):
    r_history = r_model.evaluate(r_test_input_data, r_test_output_data, verbose=2)

    return r_history


# ============================================
# Train VQVAE
# this will need to turn into the train_VAE function, which I accidently "fixed" when I started getting issues... when using it to train the VQVAE... :(
# ============================================
# def train_VQVAE(epoch, loader, model, optimizer, device, latent_loss_weight, train):
#     '''
#     params: epoch, loader, model, optimizer, device
#     checkpoint gets saved to "run_stats.pyt"
#     '''
#     iterator = tqdm(range(len(loader)//32), total=len(loader)//32)
#     sample_size = 25

#     mse_sum = 0
#     mse_n = 0
#     params = count_parameters(model)
    
#     x = torch.linspace(-1, 1, 32)
#     y = torch.linspace(-1, 1, 32)
    
#     def circular_thing(x, y):
#         return x**2+y**2
    


#     for i in iterator:
#         model.zero_grad()
#         if train:
#             images, attributes = loader.get_train_batch()
#         else:
#             images, attributes = loader.get_val_batch()
#         bool_mask = images>0
# #         RMS_width(images)
#         out, segmentation_mask, latent_loss = model(images, attributes)
#         r = (out+1e-3)/(images+1e-3)
#         SSE = -torch.log(2 * r / (r ** 2 + 1))
# #         recon_loss = (polar_loss_weights*SSE)[bool_mask]/bool_mask.float().sum()
#         recon_loss = (polar_loss_weights*SSE)/(32**2)
# #         BCE_loss = polar_loss_weights*F.binary_cross_entropy(segmentation_mask, bool_mask.float(), reduction='none')
# #         BCE_loss = BCE_loss.mean()
        
#         latent_loss = latent_loss.mean()
#         loss = recon_loss.mean() + latent_loss_weight * latent_loss # + BCE_loss
#         loss.backward()

#         optimizer.step()

#         mse_sum += recon_loss.sum().item() * images.shape[0]
#         mse_n += images.shape[0]

#         lr = optimizer.param_groups[0]["lr"]

#         iterator.set_description(
#             (
#                 f"epoch: {epoch + 1}; mse: {recon_loss.sum().item():.5f}; "
#                 f"latent: {latent_loss.item():.3f}; "#BCE loss: {BCE_loss.item():.3f} "
#                 f"lr: {lr:.5f}"
#             )
#         )

#         if (i % 300 == 0) & train:
#             model.eval()

#             sample = images[:sample_size]

#             with torch.no_grad():
#                 out, bool_out, _ = model(sample, attributes[:sample_size])
# #                 out = (bool_out>0.5).float()*out
#             utils.save_image(
#                 torch.cat([sample, out], 0),
#                 f"samples/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg",
#                 nrow=sample_size,
#                 normalize=True,
#                 range=(-1, 1),
#             )
            
#             truth = sample[0, 0][1:-1, 1:-1].cpu().numpy()
#             recon = out[0, 0][1:-1, 1:-1].cpu().numpy()
            
#             plot_3d_bar(truth, 30, 'Ground Truth (log scale)', f"samples/truth_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg")
#             plot_3d_bar(recon, 30, 'Reconstruction (log scale)', f"samples/recon_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg")
#             plot_3d_bar(np.abs(truth-recon), 30, 'Error (log scale)', f"samples/error_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg")
            
            
#             torch.save({
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#             }, 'run_stats.pyt')
#             model.train()

#         ret = {'Metric: Latent Loss': latent_loss.item(),
# #                'Metric: BCE loss':BCE_loss.item(),
#                'Metric: Reconstruction Loss': recon_loss.sum().item(),
#                'Parameter: Parameters': params,
#                'Artifact': 'run_stats.pyt'}
        
        
#         yield ret

# =================
# Train VAE
# =================
train_dict = {'Metric: Latent Loss': 0,
       'Metric: Reconstruction Loss': 0,
       'Metric: BCE loss':0,
       'Metric: Longitudual cluster asymmetry':0,
       'Metric: Transverse cluster asymmetry':0,
       'Metric: RMS width':0,
       'Metric: Momentum loss':0,
       'Metric: px/pz loss':0,
       'Metric: py/pz loss':0,
       'Metric: mse loss':0,
       'Parameter: Parameters': 0,
       'Artifact': 'run_stats.pyt'}

def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.MSELoss()(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return BCE, KLD


def circular_thing(x, y):
    '''
    Computes distance from x=y=0.
    This was used when I tried to weight the loss contribution according to distance from the center. 
    The goal was to do a better job far from the center of the event, which was usually pretty consistent
    '''
    return x**2+y**2
        
def RMS_width(E_matrix):
    '''
    Root Mean Squared (RMS) cluster width 
    
    Source: Wrote myself, but followed the lead of Dr. Fedor Ratnikov.
    This equation should be differentiable 
    '''
    y = x = torch.arange(-16, 16, device=E_matrix.device, dtype=torch.float32).unsqueeze(1).unsqueeze(1).unsqueeze(1)

    x_center = torch.tensordot(x.transpose(-1, -2), E_matrix, dims=(1,2,3))
    y_center = torch.tensordot(y, E_matrix, dims=(1,2,3))

    e_dot_x = torch.dot(E_matrix, (x-x_center)**2)
    e_dot_y = torch.dot(E_matrix, (y-y_center)**2)

    return (e_dot_x + e_dot_y).sum() / E_matrix.sum()

def asymmetry(matrix, momentum, points, orthog=False):
    '''
    Cluster asymmetry. With orthog=True, it will compute transverse cluster asymmetry. 
    
    Source: Wrote myself, but followed the lead of Dr. Fedor Ratnikov.
    This equation should be differentiable 
    '''
    matrix = matrix.squeeze()
    point = points[:, :2]
    zoff = 13
    point0 = point[:, 0] + zoff*momentum[:, 0] / momentum[:, 2]
    point1 = point[:, 1] + zoff*momentum[:, 1] / momentum[:, 2]

    if orthog:
        line_func = lambda x: (x - point0) / momentum[:, 0] * momentum[:, 1] + point1 + 1e-5
    else:
        line_func = lambda x: -(x - point0) / momentum[:, 1] * momentum[:, 0] + point1 + 1e-5 

    x = torch.linspace(-14.5, 14.5, 32, device=matrix.device)
    y = torch.linspace(-14.5, 14.5, 32, device=matrix.device)

    xx, yy = torch.meshgrid(x, y)
    zz = torch.ones((matrix.size(0), 32, 32), device=matrix.device)

    for i in range(matrix.size(0)):
        idx = torch.where(yy - line_func(xx) < 0)

        if (not orthog and point[i, 1]<0):
            idx = torch.where(yy - line_func(xx) > 0)

        zz[i][idx] *= 0
    return (torch.sum(matrix * zz, dim=(1, 2)) - torch.sum(matrix * (1 - zz), dim=(1, 2))) / (torch.sum(matrix, dim=(1, 2)) + 1e-5)

def train_VAE(epoch, loader, model, optimizer, device, latent_loss_weight, train=True):
    iterator = tqdm(range(len(loader)//32), total=len(loader)//32)
    sample_size = 25

    params = count_parameters(model)
    
    x = torch.linspace(-1, 1, 32)
    y = torch.linspace(-1, 1, 32)
    
    step = len(iterator)*epoch
    for i in iterator:
        step += 1
        model.zero_grad()
        
        if train:
            images, momentum, positions = loader.get_train_batch()
        else:
            images, momentum, positions = loader.get_val_batch()
        bool_mask = images>0
        
        # we need to know the total unscaled-summed-scaled energy because that is what we will have as input during inference, not scaled-sum
        # scaled->sum != sum->scaled
        total_energy = inv_transform(images).sum(dim=(1, 2, 3), keepdim=True)
        scaled_total_energy = transform(total_energy)
        # convert to percentages
        images = 100*images / images.sum(dim=(1, 2, 3), keepdim=True)
        
        out, pred_props, segmentation_mask, latent_loss = model(images)
        
        out = out*bool_mask
        out = 100*out / out.sum(dim=(1, 2, 3), keepdim=True)
#         out = out*bool_mask / out.sum(dim=(1, 2, 3), keepdim=True)

        # used for computing loss with total energy vs percents
#         out = scaled_total_energy*normalized
#         normalizer = lambda x: x/x.sum(-1).sum(-1).sum(-1)
#         (scaled_total_energy*normalized).sum(-1).sum(-1).sum(-1)
    
#         print(normalizer(out).sum(-1).sum(-1).sum(-1))
#         print(normalizer(out).sum(-1).sum(-1).sum(-1).shape)
#         print(scaled_total_energy.shape)
#         out = scaled_total_energy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*normalizer(out)
        
#         for i, j in zip(scaled_total_energy, (scaled_total_energy*normalized).sum(-1).sum(-1).sum(-1)):
#             print(i, j)
        pred_positions, pred_momentum = pred_props[:, :2], pred_props[:, 2:]
        
        # Compute RMS width difference
#         rms_width_truth = RMS_width(images)
#         rms_width_pred = RMS_width(out)
#         rms_width_loss = F.mse_loss(rms_width_pred, rms_width_truth)
        
        # Compute longitudual cluster asymmetry
        asymmetry_truth = asymmetry(images, momentum, positions)
        asymmetry_pred = asymmetry(out, pred_momentum, pred_positions)
        longitudual_asymmetry_loss = F.mse_loss(asymmetry_pred, asymmetry_truth)
        
        # Compute transverse cluster asymmetry
        asymmetry_truth = asymmetry(images, momentum, positions, orthog=True)
        asymmetry_pred = asymmetry(out, pred_momentum, pred_positions, orthog=True)
        transverse_asymmetry_loss = F.mse_loss(asymmetry_pred, asymmetry_truth)
        
        # compute MSE loss for momentum
        momentum_loss = F.mse_loss(pred_momentum, momentum)
        
        # momentum fraction loss - px/pz
        pxpz_loss = F.mse_loss(pred_momentum[:,0] / pred_momentum[:,2], momentum[:,0] / momentum[:,2])
        
        # momentum fraction loss - py/pz
        pypz_loss = F.mse_loss(pred_momentum[:,1] / pred_momentum[:,2], momentum[:,1] / momentum[:,2])
        
        polar_loss_weights = torch.exp(circular_thing(x[:, None], y[None, :]).unsqueeze(1).unsqueeze(1).to(device))
#         SSE = polar_loss_weights*torch.abs(torch.log(out-images))
        r = (out+1e-3)/(images+1e-3)
        SSE = -torch.log(2 * r / (r ** 2 + 1))
#         recon_loss = (polar_loss_weights*SSE)[bool_mask]/bool_mask.float().sum()
        recon_loss = SSE[bool_mask]/bool_mask.float().sum()
#         recon_loss_percents = F.mse_loss(out, images)
        mse = F.mse_loss(out[bool_mask], images[bool_mask])
#         recon_loss = polar_loss_weights*(((out-images)**2))/(32**2)
        BCE_loss = polar_loss_weights*F.binary_cross_entropy(segmentation_mask, bool_mask.float(), reduction='none')
        BCE_loss = BCE_loss.mean()
        latent_loss = latent_loss.mean()
        loss = \
            3. * recon_loss.sum() + \
            1e-2 * momentum_loss.mean() 
#             1000 * recon_loss_percents.sum() + \
#             latent_loss_weight * latent_loss + \
#             mse + \
#             3. * BCE_loss + \
#             1e-3 * longitudual_asymmetry_loss.mean() 
#             1e-4 * 5*transverse_asymmetry_loss.mean() 
#             1. * pxpz_loss.mean() + \
#             1. * pypz_loss.mean() + \
#             +  1. * rms_width_loss.mean() /
            
        train_dict['Metric: Latent Loss'] += latent_loss.mean().item()
        train_dict['Metric: Reconstruction Loss'] += 3.*recon_loss.sum().item()
#         train_dict['Metric: Reconstruction Loss'] += 1000.*recon_loss_percents.sum().item()
        train_dict['Metric: BCE loss'] += 3.*BCE_loss.mean().item()
        train_dict['Metric: Longitudual cluster asymmetry'] += 5*longitudual_asymmetry_loss.mean().item()
        train_dict['Metric: Transverse cluster asymmetry'] += 5*transverse_asymmetry_loss.mean().item()
#         train_dict['Metric: RMS width'] += .mean().item()
        train_dict['Metric: Momentum loss'] += 1e-2*momentum_loss.mean().item()
        train_dict['Metric: px/pz loss'] += pxpz_loss.mean().item()
        train_dict['Metric: py/pz loss'] += pypz_loss.mean().item()
        train_dict['Metric: mse loss'] += mse.mean().item()
        
        loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]

        iterator.set_description(
            (
                f"epoch: {epoch + 1}; mse: {recon_loss.sum().item():.5f}; "
#                 f"epoch: {epoch + 1}; mse pct: {1000*recon_loss_percents.sum().item():.5f}; "
                f"latent: {latent_loss.item():.3f}; BCE loss: {BCE_loss.item():.3f}; Momentum: {1e-2*momentum_loss.item():.3f} "
                f"lr: {lr:.5f}"
                f"mse: {mse.item():.5f}"
            )
        )

        if ((i+1) % 50 == 0) & train:
            for k, v in train_dict.items():
                if isinstance(v, float):
                    train_dict[k] /= 50
            save_to_mlflow(train_dict, step)
            for k, v in train_dict.items():
                if isinstance(v, float):
                    train_dict[k] = 0

            
        if ((i+1) % 300 == 0) & train:
            model.eval()

            sample = images[:sample_size]

            with torch.no_grad():
#                 total_energy = inv_transform(sample).sum(dim=(1, 2, 3), keepdim=True)
#                 scaled_total_energy = transform(total_energy)
#                 sample = sample/sample.sum(dim=(1, 2, 3), keepdim=True)
                out, pred_props, bool_out, _ = model(sample)
                out = 100*out / out.sum(dim=(1, 2, 3), keepdim=True)
#                 normalized_out = out/out.sum(dim=(1, 2, 3), keepdim=True)
#                 out = scaled_total_energy*out/100
#                 out = scaled_total_energy*normalized_out
                # apply mask
#                 out = (bool_out>torch.rand_like(bool_out)).float()*out
            utils.save_image(
                torch.cat([sample, out], 0),
                f"samples/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg",
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )
            
            truth = sample[0, 0][1:-1, 1:-1].cpu().numpy()
            recon = out[0, 0][1:-1, 1:-1].cpu().numpy()
            
            plot_3d_bar(truth, 30, f'Ground Truth - sum/100: {truth.sum()/100}', f"samples/truth_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg")
            plot_3d_bar(recon, 30, f'Reconstruction - sum/100: {recon.sum()/100}', f"samples/recon_{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg")
            
            
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'run_stats.pyt')
            model.train()

        yield train_dict
# def train_VAE(epoch, loader, model, optimizer, device):
#     '''
# 	params: epoch, loader, model, optimizer, device
# 	checkpoint gets saved to "run_stats.pyt"
# 	'''
#     loader = tqdm(loader)

#     loss_sum = 0
#     # params = count_parameters(model)
#     for i, img in enumerate(loader):
#         model.zero_grad()

#         img = img.to(device)
#         recon_image, mu, logvar = model(img)
#         MSE, KL = vae_loss(recon_image, img, mu, logvar)
#         MSE = MSE.to(device)
#         KL = KL.to(device)
#         loss = MSE+0.0007*KL
#         loss.backward()
#         optimizer.step()

#         loss_sum += loss.item() * img.shape[0]

#         # lr = optimizer.param_groups[0]["lr"]

#         loader.set_description((f"epoch: {epoch + 1}; MSE loss: {MSE.item():.5f}; KL loss: {KL.item():.5f} "))

#         if i % 300 == 0:
#             model.eval()
#             sample_size = 10
#             sample = img[:sample_size]

#             with torch.no_grad():
#                 out, _, _ = model(sample)
#                 print('output max', out.max().item())
#                 print('sample max', sample.max().item())
#                 print('output mean', out.mean().item())
#                 print('truth mean', sample.mean().item())
#             utils.save_image(
#                 torch.cat([sample, out], 0),
#                 f"samples/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.jpg",
#                 nrow=sample_size,
#                 # normalize=True,
#                 # range=(-1, 1),
#             )

#             torch.save({
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#             }, 'run_stats.pyt')
#             model.train()

#         ret = {'Metric: Loss': loss.item(),
#                # 'Parameter: Parameters': params,
#                'Artifact': 'run_stats.pyt'}
#         yield ret
