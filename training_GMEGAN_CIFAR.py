#!/usr/bin/python
""" 
    Run with the following command
    python training_GMEGAN.py --lr=1e-5
    generating from images from CIFAR-10 Dataset
    same as the algorithm in the paper
"""

import argparse
import time
import os
import sys
import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.autograd as autograd

from transportmodules.transports20 import *

matplotlib.use('Agg')

parser = argparse.ArgumentParser()
# general arguments
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--num_iter', type=int, default=5000)
# arguments to choose dataset (mnist, cifar, cifar-gray etc.)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--plot_every', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--starting_epoch', type=int, default=0)
parser.add_argument('--saving', type=str, default='0')
parser.add_argument('--module', type=str, default='18')
# arguments for Gaussian mixture application
parser.add_argument('--lr', type=float, default=1e-5)

args = parser.parse_args()
print(args)

# system preferences
torch.set_default_dtype(torch.float)
seed = np.random.randint(100)
np.random.seed(seed)
torch.manual_seed(seed)

save_fig_path  = f'out_cifar_{args.saving}'
save_data_path = f'data_cifar_{args.saving}'
os.makedirs(save_fig_path, exist_ok=True) ;print(f"saving images in {save_fig_path}")
os.makedirs(save_data_path, exist_ok=True);print(f"saving data in {save_data_path}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

if cuda:
    print("XXX GPU is available XXX")
else:
    print("XXX GPU is not available XXX")

# settings
n = 5000 # data size
z_dim = 300
img_size = 32 # size of the image 32 x 32
n_channels = 3 # number of channels 3 if color images
img_shape = (n_channels, img_size, img_size)
batch_size = args.batch_size


def calculate_fid(real_mu_sigma, i, generated_embeddings):
    generated_embeddings = generated_embeddings.reshape((generated_embeddings.shape[0],-1))

    mu1, sigma1 = real_mu_sigma['mu'][i], real_mu_sigma['sigma'][i]
    mu2, sigma2 = torch.mean(generated_embeddings, dim=0), torch.cov(generated_embeddings.t())

    # Calculate sum squared difference between means
    ssdiff = torch.sum((mu1 - mu2)**2.0)
    
    # Calculate square root of the product between covariances
    covmean = torch.sqrt(torch.mul(sigma1, sigma2))
    
    # Convert to real part if complex
    if covmean.is_complex():
        covmean = covmean.real
    
    # Calculate FID score
    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid.item()  # Convert to scalar value

def saving_models(save_data_path, epoch):
    torch.save(model['T'].state_dict(), f"{save_data_path}/{epoch}-T.pt")
    torch.save(model['G'].state_dict(), f"{save_data_path}/{epoch}-G.pt")
    torch.save(model['psi'].state_dict(), f"{save_data_path}/{epoch}-psi.pt")
        
def loading_models(models, Titlelist, save_data_path, epoch):
    for title in Titlelist:
        try:
            models[title]['T'].load_state_dict(torch.load(f"{save_data_path}/{epoch}-{title}-T.pt"))
            models[title]['G'].load_state_dict(torch.load(f"{save_data_path}/{epoch}-{title}-G.pt"))
        except:
            pass

def get_latent_samples(shape, title='None'):
    return torch.randn(shape,device=device)

def compute_GME_cost(x, Tx, eta=lambda x: x):
    """
        eta = lambda x: x
            = lambda x: 1.0/(1.0+x)
            = lambda x: log(1.0+x)
    """
    n      = x.shape[0]
    d_x    = np.prod(x.shape[1:])
    d_T    = np.prod(Tx.shape[1:])
    xy     = ((x.view((n, 1, d_x)) - x.view((1, n, d_x)))**2).sum(dim=2)
    Txy    = ((Tx.view((n, 1, d_T)) - Tx.view((1, n, d_T)))**2).sum(dim=2) * 9
    Axy    = eta(xy)
    ATxy   = eta(Txy)
    return F.mse_loss(Axy, ATxy)

def iterate_GMEGAN(model, x, c_T=1, c_G=1, c_R=0.5, k=1.0, eta=lambda x: x):
    T   = model['T'].forward_T
    G   = model['G']
    psi = model['psi']
    optG   = model['optG']
    optT   = model['optT']
    optpsi = model['optpsi']

    n = x.shape[0]

    def compute_disc_loss(psi, Gz, x):
        return psi(Gz).mean() - psi(x).mean() 

    # ---------------------
    #  Train S and T
    # ---------------------
    optT.zero_grad()
    z = get_latent_samples((n,z_dim))
    Tx = T(x)
    Gz = G(z)
    T_loss  = compute_GME_cost(x, Tx, eta=eta)
    R_loss = F.mse_loss(z,T(Gz))
    psi_loss = compute_disc_loss(psi,  Gz,  x)
    loss = R_loss * c_R + psi_loss + T_loss * c_T
    loss.backward()
    optT.step()

    optG.zero_grad()
    z = get_latent_samples((n,z_dim))
    Tx = T(x).detach()
    Gz = G(z)
    R_loss = F.mse_loss(z,T(Gz))
    psi_loss = compute_disc_loss(psi,  Gz,  x)
    G_loss = F.mse_loss(x,G(Tx))
    loss = R_loss * c_R + G_loss * c_G + psi_loss
    loss.backward()
    optG.step()

    # ---------------------
    #  Train psi
    # ---------------------
    optpsi.zero_grad()
    z = get_latent_samples((n,z_dim))
    Gz = G(z)
    psi_loss = compute_disc_loss(psi,  Gz,  x)
    x = x.requires_grad_(True)
    # Real images
    real_validity = psi(x)
    # Fake images
    fake_validity = psi(Gz)
    # Compute W-div gradient penalty
    real_grad_out  = Tensor(np.ones((x.shape[0], 1))).requires_grad_(False)
    real_grad      = autograd.grad( real_validity, x, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True )[0]
    real_grad_norm = real_grad.view(real_grad.shape[0], -1).pow(2).sum(1) 
    fake_grad_out  = Tensor(np.ones((Gz.shape[0], 1))).requires_grad_(False)
    fake_grad      = autograd.grad( fake_validity, Gz, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True )[0]
    fake_grad_norm = fake_grad.view(fake_grad.shape[0], -1).pow(2).sum(1)
    div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
    loss  = - psi_loss + div_gp
    loss.backward()
    optpsi.step()

def preprocessing_data(sample_size = 1000):
    """
        Getting ready with the data.
        Will return dataloader, x_val, label_val
    """
    # Configure data loader
    data_dir = '../data/cifar'
    transform = transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])]
            )
    os.makedirs(data_dir, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            data_dir,
            download=True,
            transform=transform,
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    dataloader2 = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            data_dir,
            download=False,
            transform=transform,
        ),
        batch_size=sample_size,
        shuffle=True,
    )
    for _, (imgs, labels) in enumerate(dataloader2):
        x_val = imgs.to(device)
        label_val = labels
        break
    
    def calculate_fid_compute_mu_sigma(real_embeddings):
        real_embeddings      = real_embeddings.reshape((real_embeddings.shape[0],-1))
        mu, sigma = torch.mean(real_embeddings, dim=0), torch.cov(real_embeddings.t())
        return mu, sigma

    # compute mu and sigma of the real data in the beginning
    real_mu_sigma = {'mu':[], 'sigma':[]}
    for i, (imgs, _) in enumerate(dataloader2):
        x = imgs.to(device) # Images
        mu, sigma = calculate_fid_compute_mu_sigma(x) # compute fid multiple times to get the average fid score
        real_mu_sigma['mu'].append(mu)
        real_mu_sigma['sigma'].append(sigma)
    del dataloader2
    return dataloader, x_val, label_val, real_mu_sigma

# --------------------------------------------------------------------------------
# Getting the data
sample_size = 1000
dataloader, x_val, label_val, real_mu_sigma = preprocessing_data(sample_size = sample_size)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

Titlelist = ['GMEGAN']

# create optimiser
lr = args.lr
b1 = 0.5
b2 = 0.999

model = {}
model['G']      = TransportG(z_dim=z_dim, output_shape=img_shape).to(device)
model['T']      = TransportT(input_shape=img_shape, z_dim=z_dim).to(device)
model['psi']    = TransportT(input_shape=img_shape, z_dim=z_dim).to(device)
model['optG']   = torch.optim.Adam(model['G'].parameters(),   lr=2*lr, betas=(b1, b2))
model['optT']   = torch.optim.Adam(model['T'].parameters(),   lr=lr, betas=(b1, b2))
model['optpsi'] = torch.optim.Adam(model['psi'].parameters(), lr=lr, betas=(b1, b2))

# ------------------------------------------------------------------------------------------------
# Training
pbar = tqdm.tqdm(range(args.starting_epoch, args.num_epochs), desc="Outer iteration")
for epoch in pbar:
    for i, (imgs, labels) in enumerate(tqdm.tqdm(dataloader, desc=f"epoch:{epoch}", leave=False)): 
        x = imgs.to(device)
        iterate_GMEGAN(model, x, c_T=10, c_G=1, c_R=0.5, k=1.0, eta=lambda x: torch.log(1.0+x))
    if epoch % 50 == 0:
        with torch.no_grad():
            pbar.write(f"Saving at epoch: {epoch}")
            # Compute FID score
            G = model['G']
            N_fid = 5
            fid_arr_tmp = np.zeros(N_fid)
            for i in range(N_fid):
                z = get_latent_samples((sample_size, z_dim))
                fid_arr_tmp[i] = calculate_fid(real_mu_sigma, i, G(z).detach()) # compute fid multiple times to get the average fid score
            fid = np.mean(fid_arr_tmp)
            pbar.write(f"FID: {fid:0.2f}")
            # Saving the model
            saving_models(model, save_data_path, epoch)