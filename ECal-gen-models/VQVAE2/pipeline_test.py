import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from VQVAE import VQVAE
from train import train
from utilities import start_mlflow_experiment, Params, save_to_mlflow, count_parameters, load_full_state, select_gpu

from tqdm import tqdm
import mlflow

# device = select_gpu(1)
device = 'cuda:1'
args = Params(32, 10, 4e-4, 256, device)

start_mlflow_experiment('Vector Quantized Variational Autoencoder', 'lane-finder')


transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

dataset = datasets.ImageFolder('/share/lazy/will/ConstrastiveLoss/Imgs/color_images/train/', transform=transform)
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory = True)

model = VQVAE(channel=128).to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

load_full_state(model, optimizer, '/share/lazy/will/ConstrastiveLoss/Logs/0/f5439ae7f4f240ceac163b01614c49a6/artifacts/run_stats.pyt')

run_name = 'overnight run'
with mlflow.start_run(run_name = run_name) as run:

    for epoch in range(args.epoch):
        results = train(epoch, loader, model, optimizer, args.device)
        for Dict in results:
            save_to_mlflow(Dict, args)
