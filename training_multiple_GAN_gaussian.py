#!/usr/bin/python

""" 
    Run with the following command
    python w2gan-generate-different-dim.py --modes 3d_4mode --num_iter 10000 --l1reg --advsy

    generating from 2,3d to 2,3d.
    same as the algorithm in the paper 
"""

import argparse
import time
import random
from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tqdm

import os
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

latent_r = 1.8

parser = argparse.ArgumentParser()

# general arguments
parser.add_argument('--num_iter', type=int, default=1000000)
parser.add_argument('--sigma', type=float, default=0.15)
parser.add_argument('--modes', type=int, default=4)
parser.add_argument('--plot_every', type=int, default=1000)
parser.add_argument('--input_dim', type=int, default=100)

# arguments for Gaussian mixture application
parser.add_argument('--lr', type=float, default=1e-3)

opt = parser.parse_args()
print(opt)

def subplot_model(ind, ax, gs, T, S, x, z, latent_val, label_val, z_val_sizes, z_val_labels, title, centers):
    Tx = T(x, feature=True)
    Sz = S(z)
    Sz0 = S(latent_val)

    if cuda:
        z = z.cpu()
        Tx = Tx.cpu()
        x = x .cpu()
        Sz = Sz.cpu()
        Sz0 = Sz0.cpu()
        centers = centers.cpu()
        latent_val = latent_val.cpu()
    
    ax = fig.add_subplot(gs[0,ind])
    ax.scatter(Tx[:,0],Tx[:,1],c=label_val,alpha=0.5,cmap='tab20',vmin=0, vmax=max(centerind2label.values()))
    ax.set_title(f'$T_\\#\\mu$ from {title}')
    ax.set_aspect('equal')
    ax = fig.add_subplot(gs[1,ind])
    ax.scatter(x[:,0],x[:,1],alpha=0.5,marker='o',c='black') # c=label_val,
    ax.scatter(Sz[:,0],Sz[:,1],s=z_val_sizes,alpha=0.4,marker='x',c=z_val_labels,cmap='gist_rainbow')
    # ax.set_title(f"$S_\\#\\nu$ from {title}", fontsize=16)
    if title == "GMEGAN":
        ax.set_title(f"GMEGAN (ours)", fontsize=17)
    else:
        ax.set_title(f"{title}", fontsize=17)
    # ax.set_aspect('equal')
    ax.set_xlim([-3.3,4.7]); ax.set_ylim([-3.2,4.7])

    z = latent_val.numpy()
    
    ax = fig.add_subplot(gs[2,ind])
    colors = np.zeros(z.shape[0])
    cc = ((centers.reshape((1, centers.shape[0], input_dim)) - Sz0.reshape((Sz0.shape[0], 1, input_dim)))**2).mean(dim=2)
    cc = cc.numpy()
    colors = np.argmin(cc, axis=1)
    colors = [centerind2label[f'{i}'] for i in colors]
    ax.scatter(z[:,0],z[:,1],alpha=0.5,marker='o',c=colors,cmap='tab20',vmin=0, vmax=max(centerind2label.values())) # c=label_val,
    ax.set_xlim([-latent_r,latent_r])
    ax.set_ylim([-latent_r,latent_r])
    # ax.set_title(f"$\\nu$ from {title}", fontsize=20)
    ax.set_aspect('equal')
sigma = opt.sigma

def create_dataset(sample_size, n_mode=4, input_dim=3):
    scale = 0.8
    dim = input_dim
    # centers = [[0]*dim]
    centers = []
    centerind2label = {}
    ind = 0
    r = opt.sigma * 1.1

    for i in range(n_mode):
        for j in [3.0]:
            x = np.cos(2.0*np.pi*i/n_mode)
            y = np.sin(2.0*np.pi*i/n_mode)
            point = np.random.randn(dim) * 0.0
            point[0] = x * r * j + 0.5
            point[1] = y * r * j + 0.5
            centers.append(point)
            centerind2label[f'{ind}'] = ind
            ind += 1
    
    n_mode2 = n_mode*2
    for i in range(n_mode2):
        for j in [9.0]:
            x = np.cos(2.0*np.pi*i/n_mode2)
            y = np.sin(2.0*np.pi*i/n_mode2)
            point = np.random.randn(dim) * 0.0
            point[0] = x * r * j + 0.5
            point[1] = y * r * j + 0.5
            centers.append(point)
            centerind2label[f'{ind}'] = i + n_mode2
            ind += 1

    centers = np.array(centers)
    dataset = []
    y = []
    sigma_const = 0.1
    for i in range(sample_size):
        point = np.random.randn(dim)*opt.sigma
        point[2:] *= sigma_const
        index  = random.randint(0, len(centers)-1)
        point += centers[index]
        dataset.append(point)
        y.append(centerind2label[f'{index}'])
    x = np.array(dataset, dtype='float32')
    
    centers = np.array(centers)

    # x = (x -np.min(x,axis=0)) /(np.max(x,axis=0) - np.min(x,axis=0)) * 2.0 - 1.0
    # print(x.shape)
    return x, np.array(y), centers, centerind2label

def compute_S_loss2(Tx,STx):
    n     = Tx.shape[0]
    d_T   = np.prod(Tx.shape[1:])
    d_ST  = np.prod(STx.shape[1:])
    Txy   = ((Tx.view((n, 1, d_T)) - Tx.view((1, n, d_T)))**2).sum(dim=2)
    STxy  = ((STx.view((n, 1, d_ST)) - STx.view((1, n, d_ST)))**2).sum(dim=2)
    AT    = 1.0/(1.0+Txy/2)
    AST   = 1.0/(1.0+STxy/2)
    PT  = AT /torch.sum(AT,  dim = 1, keepdims=True)
    PST = AST/torch.sum(AST, dim = 1, keepdims=True)
    P    = PT * torch.log(PT/PST)
    return P.mean()

def compute_GME_cost(x, Tx, eta=lambda x: x):
    n      = x.shape[0]
    d_x    = np.prod(x.shape[1:])
    d_T    = np.prod(Tx.shape[1:])
    xy     = ((x.view((n, 1, d_x)) - x.view((1, n, d_x)))**2).sum(dim=2)
    Txy    = ((Tx.view((n, 1, d_T)) - Tx.view((1, n, d_T)))**2).sum(dim=2)
    Axy    = eta(xy)
    ATxy   = eta(Txy)
    loss = ((Axy - ATxy)**2).mean()

    return loss

def compute_lap_loss(x, Tx):
    n      = x.shape[0]
    d_x    = np.prod(x.shape[1:])
    d_T    = np.prod(Tx.shape[1:])
    xy     = ((x.view((n, 1, d_x)) - x.view((1, n, d_x)))**2).sum(dim=2)
    Txy    = ((Tx.view((n, 1, d_T)) - Tx.view((1, n, d_T)))**2).sum(dim=2)
    eta = lambda x: torch.exp(-x)
    return (eta(xy) * Txy).mean()


def compute_laplacian3(x, Tx):
    n      = x.shape[0]
    d_x    = np.prod(x.shape[1:])
    d_T    = np.prod(Tx.shape[1:])
    xy     = ((x.view((n, 1, d_x)) - x.view((1, n, d_x)))**2).sum(dim=2)
    Txy    = ((Tx.view((n, 1, d_T)) - Tx.view((1, n, d_T)))**2).sum(dim=2)
    return (xy * 1.0/(1.0 + Txy)).mean()

def compute_RGM_cost(x,Tx,z,Sz):
    n      = x.shape[0]
    d_x    = np.prod(x.shape[1:])
    d_z    = np.prod(z.shape[1:])
    Ax     = (( x.view((n, 1, d_x)) - Sz.view((1, n, d_x)))**2).sum(dim=2)
    Az     = ((Tx.view((n, 1, d_z)) -  z.view((1, n, d_z)))**2).sum(dim=2)
    return ((Ax-Az)**2).mean()

cuda   = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    print("GPU is available")
else:
    print("GPU not available.")

# plotting preferences
matplotlib.rcParams['font.size'] = 10

# system preferences
# torch.set_default_dtype(torch.float)
seed = np.random.randint(100)
np.random.seed(seed)
torch.manual_seed(seed)


save_fig_path = 'out_gaussian'
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)
print(f"saving images in {save_fig_path}")

# settings
n = 10000 # data size
batch_size = 16

input_dim = opt.input_dim
z_dim = 2
plot_every = opt.plot_every
niter = 10
epsilon = 0.01
ngen = 5
beta = 1
stop_adversary = opt.num_iter
train_iter = opt.num_iter
modes = opt.modes

class Rinv(torch.nn.Module):
    def __init__(self, x_dim = 100, z_dim = 100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, z_dim),
        )
    def forward(self, x):
        x  = self.model(x)
        return x  
    
class Discriminator(torch.nn.Module):
    def __init__(self, x_dim = 100):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, z_dim),
        )
    def forward(self, x):
        x  = self.model(x)
        return x  


class VEEGAN_D(torch.nn.Module):
    def __init__(self, x_dim = 100, z_dim = 100):
        super(VEEGAN_D, self).__init__()
        self.model1 = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, z_dim),
            # nn.Sigmoid(),
        )
        self.model2 = nn.Sequential(
            nn.Linear(z_dim*2, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, 1),
        )
        
    def forward(self, x, z):
        x = self.model1(x)
        xz = torch.cat((x,z), 1)
        return self.model2(xz)
    
sigmoid = nn.Sigmoid()

class TransportT(torch.nn.Module):
    def __init__(self, x_dim = 100, z_dim = 100):
        super(TransportT, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(x_dim, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, 128),
        )
        self.end_T = nn.Sequential(
            nn.Linear(128, z_dim),
        )
        self.phi = nn.Sequential(
            nn.LeakyReLU(0.2,True),
            nn.Linear(z_dim, 1),
        )
        self.output = nn.Sequential(
            nn.Tanh(),
        )
        
        self.sig = nn.Sequential(
            nn.Sigmoid(),
        )

        # latent mean and variance 
        self.mean_layer = nn.Sequential(nn.Linear(128, z_dim))
        self.logvar_layer = nn.Sequential(nn.Linear(128, z_dim))

    def forward(self, x, feature=False):
        Tx = self.end_T(self.model(x))
        if feature:
            return Tx
        return self.phi(Tx)
    
    def forward_feature(self, x):
        return self.model(x)
    
    def forward_T(self, x):
        return self.end_T(self.model(x))

    def disc(self, x):
        return self.phi(self.model(x))

    def gan(self,x):
        x = self.forward(x)
        return self.sig(x)
    
    def get_mean_and_var(self, x):
        Tx = self.model(x)
        return self.mean_layer(Tx), self.logvar_layer(Tx)

class TransportG(torch.nn.Module):
    def __init__(self, z_dim = 100, x_dim = 100):
        super(TransportG, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, 128),
            # nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, x_dim),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Reconstructor(torch.nn.Module):
    def __init__(self, z_dim = 2):
        super(Reconstructor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2,True),
            nn.Linear(128, z_dim),
        )

    def forward(self, x):
        x = self.model(x)
        return x


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples

    alpha = torch.rand((real_samples.shape[0], 1), device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)

    fake = torch.ones((real_samples.shape[0], 1), device=device).requires_grad_(False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def get_latent_samples(shape):
    return torch.randn(shape, device=device)

# data simulation
data, y, centers, centerind2label = create_dataset(n, opt.modes, input_dim)


centers = torch.tensor(centers, device=device)

sample_size = 1000

real = data[:sample_size]
label_val = y[:sample_size]

lr =opt.lr
b1 = 0.5
b2 = 0.999

GWlist = ['GMEGAN']
titlelist =  [*GWlist, "GAN", "WGAN", "WGP", 'WDIV', "OTM", 'VAEGAN', 'VEEGAN']
timelist  = {title: 0  for title in titlelist}
models    = {title: {} for title in titlelist}

for key in titlelist:
    if 'GMEGAN' in key:
        models[key]['T']    = TransportT(x_dim=input_dim, z_dim=z_dim).to(device)
        models[key]['S']    = TransportG(z_dim=z_dim,     x_dim=input_dim).to(device)
        models[key]['psi']  = TransportT(x_dim=input_dim, z_dim=z_dim).to(device)
        models[key]['Rinv'] = Rinv(x_dim=z_dim, z_dim=z_dim).to(device)
        models[key]['phi']  = Discriminator(x_dim=z_dim).to(device)
        models[key]['optG'] = torch.optim.Adam([*models[key]['S'].parameters(),*models[key]['T'].parameters(),*models[key]['Rinv'].parameters()], lr=1*lr, betas=(b1, b2))
        models[key]['optT'] = torch.optim.Adam(models[key]['T'].parameters(), lr=lr, betas=(b1, b2))
        models[key]['optpsi'] = torch.optim.Adam([*models[key]['psi'].parameters()],   lr=lr, betas=(b1, b2))
    elif 'VAEGAN' in key:
        models[key]['T']    = TransportT(x_dim=input_dim, z_dim=z_dim).to(device)
        models[key]['S']    = TransportG(z_dim=2,     x_dim=input_dim).to(device)
        models[key]['psi']  = TransportT(x_dim=input_dim, z_dim=z_dim).to(device)
        models[key]['optT'] = torch.optim.Adam(models[key]['T'].parameters(),       lr=lr, betas=(b1, b2))
        models[key]['optS'] = torch.optim.Adam(models[key]['S'].parameters(),       lr=1*lr, betas=(b1, b2))
        models[key]['optpsi'] = torch.optim.Adam(models[key]['psi'].parameters(),   lr=lr, betas=(b1, b2))
    elif 'VEEGAN' in key:
        models[key]['T']    = TransportT(x_dim=input_dim, z_dim=z_dim).to(device)
        models[key]['S']    = TransportG(z_dim=z_dim,     x_dim=input_dim).to(device)
        models[key]['phi']  = VEEGAN_D(x_dim=input_dim, z_dim=z_dim).to(device)
        models[key]['optT'] = torch.optim.Adam(models[key]['T'].parameters(), lr=lr, betas=(b1, b2))
        models[key]['optS'] = torch.optim.Adam(models[key]['S'].parameters(), lr=1*lr, betas=(b1, b2))
        models[key]['optphi'] = torch.optim.Adam(models[key]['phi'].parameters(), lr=lr, betas=(b1, b2))
    else:
        models[key]['T']    = TransportT(x_dim=input_dim, z_dim=z_dim).to(device)
        models[key]['S']    = TransportG(z_dim=z_dim,     x_dim=input_dim).to(device)
        models[key]['optT'] = torch.optim.Adam(models[key]['T'].parameters(),       lr=lr, betas=(b1, b2))
        models[key]['optS'] = torch.optim.Adam(models[key]['S'].parameters(),       lr=1*lr, betas=(b1, b2))

## OTM
# Embeddings
# Q = lambda x: F.interpolate(x.reshape(-1, z_dim), input_dim, mode='linear').detach() 
def Q(x):
    Qx = torch.zeros((x.shape[0],input_dim)).to(device)
    Qx[:,:x.shape[1]]= x
    return Qx

INV_TRANSFORM = lambda x: 0.5*x + 0.5

def Loss(psi, G, Q, z, x):
    """
        Computing loss for OTM method
        psi: discriminator T
        G: Generator
        Q: Interpolation z -> x
        z: latent variable
        x: real images
    """
    G_z = G(z)
    dot = torch.mean(Q(z)*G_z, dim=(1)).unsqueeze(dim=1)
    loss = ( dot - psi(G_z) + psi(x)).mean()
    return loss

def GradientOptimality(psi, G, Q, x):
    """ Gradient Optimality cost for potential"""
    G_x = G(x)
    G_x.requires_grad_(True)
    psi_G_x = psi(G_x)

    gradients = autograd.grad(
        outputs=psi_G_x, inputs=G_x,
        grad_outputs=torch.ones(psi_G_x.size()).to(G_x),
        create_graph=True, retain_graph=True
    )[0]
    return (gradients.mean(dim=0) - Q(x).mean(dim=0)).norm('fro')


# VAE loss
def get_x_mean_logvar(T, x):
    x = T(x,feature=True)
    mean, logvar = T.mean_layer(x), T.logvar_layer(x)
    epsilon = torch.randn_like(logvar).to(device)      
    z = mean + logvar*epsilon
    x_hat = S(z)
    return x_hat, mean, logvar
    

def loss_function_VAE(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss, KLD


# sample for plotting
z_val = get_latent_samples((sample_size, z_dim))

latent_val = torch.zeros((1, z_dim), device=device)

while latent_val.shape[0] < sample_size:
    latent_val_tmp = (torch.rand((sample_size, z_dim),device=device)-0.5)*2*latent_r
    check = ((latent_val_tmp**2).sum(1) < latent_r**2)
    latent_val = torch.cat((latent_val, latent_val_tmp[check]), 0)


latent_val = latent_val[1:,:]

z_val_labels  = z_val.norm(2,dim=1)
z_val_labels, indices = torch.sort(z_val_labels, descending=True)
z_val = z_val[indices]
z_val_labels /= torch.max(z_val_labels)
z_val_sizes = (z_val_labels)*100 + 10
z_val_labels = z_val_labels.cpu()
z_val_sizes=  z_val_sizes.cpu()

# set iterator for plot numbering
epoch = 0
T_loss_global = 1
time_error_arr = []
# Loss function for GAN
adversarial_loss = torch.nn.BCELoss()

start_time = time.time()
pbar = tqdm.tqdm(range(train_iter))
for it in pbar:
    start_idx = it * batch_size % n

    X_mb = data[start_idx:start_idx + batch_size, :]
    y_mb = y[start_idx:start_idx + batch_size]

    # get data mini batch

    x = torch.tensor(X_mb[:batch_size, :], device=device)
    # x = x.double()
    y_s = y_mb[:batch_size]

    #------------ GAN ------------
    if True:
        start_time = time.time()
        T      = models['GAN']['T']
        S      = models['GAN']['S']
        # phi    = models['GAN']['phi']
        optT   = models['GAN']['optT']
        optS   = models['GAN']['optS']
        # optphi = models['GAN']['optphi']
        # -----------------
        #  Train Generator
        # -----------------



        valid = torch.ones((x.shape[0],  1), device=device).requires_grad_(False)
        fake  = torch.zeros((x.shape[0], 1), device=device).requires_grad_(False)

        optS.zero_grad()

        # Sample noise as generator input
        z = get_latent_samples((x.shape[0],z_dim))
        # z = z.double()
        # Generate a batch of images
        gen_imgs = S(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(T.gan(gen_imgs), valid)

        g_loss.backward()
        optS.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optT.zero_grad()
        gen_imgs = S(z)

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(T.gan(x), valid)
        fake_loss = adversarial_loss(T.gan(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optT.step()

        timelist["GAN"] = time.time() - start_time


    #------------ WGAN ------------
    if True:
        start_time = time.time()
        T      = models['WGAN']['T']
        S      = models['WGAN']['S']
        # phi    = models['WGAN']['phi']
        optT   = models['WGAN']['optT']
        optS   = models['WGAN']['optS']
        # optphi = models['WGAN']['optphi']
        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(3):
            optT.zero_grad()


            z = get_latent_samples((x.shape[0],z_dim))
            # Generate a batch of images
            fake_imgs = S(z)
            # Real images
            real_validity = T(x)
            # Fake images
            fake_validity = T(fake_imgs)
            # Adversarial loss
            T_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
            T_loss.backward()
            optT.step()

        # Train the generator every n_critic steps
        # -----------------
        #  Train Generator
        # -----------------
        for _ in range(1):
            optS.zero_grad()
            # Generate a batch of images
            fake_imgs = S(z)
            # Loss measures generator's ability to fool the T
            # Train on fake images
            fake_validity = T(fake_imgs)
            S_loss = -torch.mean(fake_validity)
            S_loss.backward()
            optS.step()

        timelist["WGAN"] = time.time() - start_time

    #------------ WGAN GP ------------
    if True:
        start_time = time.time()
        T      = models['WGP']['T']
        S      = models['WGP']['S']
        optT   = models['WGP']['optT']
        optS   = models['WGP']['optS']
        # Loss weight for gradient penalty
        lambda_gp = 10
        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(2):
            optT.zero_grad()


            z = get_latent_samples((x.shape[0],z_dim))
            # Generate a batch of images
            fake_imgs = S(z)

            # Real images
            real_validity = T(x)
            # Fake images
            fake_validity = T(fake_imgs)
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(T, x.data, fake_imgs.data)
            # Adversarial loss
            T_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            T_loss.backward()
            optT.step()

        # Train the generator every n_critic steps
        for _ in range(1):

            # -----------------
            #  Train Generator
            # -----------------
            optS.zero_grad()
            # Generate a batch of images
            fake_imgs = S(z)
            # Loss measures generator's ability to fool the T
            # Train on fake images
            fake_validity = T(fake_imgs)
            S_loss = -torch.mean(fake_validity)

            S_loss.backward()
            optS.step()

        timelist['WGP'] = time.time() - start_time

    #------------ WGAN DIV ------------
    def iterate_WDIV(model):
        T      = model['T']
        S      = model['S']
        # phi    = model['phi']
        optT   = model['optT']
        optS   = model['optS']
        # optphi = model['optphi']
        # ---------------------
        #  Train Discriminator
        # ---------------------
        p=6;k=2
        x.requires_grad_(True)
        for _ in range(2):
            optT.zero_grad()

            # Sample noise as generator input


            z = get_latent_samples((x.shape[0],z_dim))

            # Generate a batch of images
            fake_imgs = S(z)

            # Real images
            real_validity = T(x)
            # Fake images
            fake_validity = T(fake_imgs)

            # Compute W-div gradient penalty
            real_grad_out = torch.ones((x.shape[0], 1), device=device).requires_grad_(False)
            real_grad = autograd.grad(
                real_validity, x, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            real_grad_norm = real_grad.view(real_grad.shape[0], -1).pow(2).sum(1) ** (p / 2)

            fake_grad_out = torch.ones((fake_imgs.shape[0], 1), device=device).requires_grad_(False)
            fake_grad = autograd.grad(
                fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            fake_grad_norm = fake_grad.view(fake_grad.shape[0], -1).pow(2).sum(1) ** (p / 2)

            div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + div_gp

            d_loss.backward()
            optT.step()

        # -----------------
        #  Train Generator
        # -----------------
        for _ in range(1):
            optS.zero_grad()
            # Generate a batch of images
            fake_imgs = S(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = T(fake_imgs)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optS.step()

    start_time = time.time()
    iterate_WDIV(models['WDIV'])
    timelist['WDIV'] = time.time() - start_time


    #------------ OTM ------------ https://github.com/LituRout/OptimalTransportModeling/blob/main/source/otm_mnist_32x32.py#L426
    def iterate_OTM(model):
        T      = model['T']
        S      = model['S']
        optT   = model['optT']
        optS   = model['optS']

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for _ in range(1):
            optT.zero_grad()


            z = get_latent_samples((x.shape[0],z_dim))
            loss = Loss(T, S, Q, z, x) 
            go_loss = GradientOptimality(T, S, Q, z)
            lam_otm = 10 # as in the github
            T_loss = loss + lam_otm * go_loss
            T_loss.backward()
            optT.step()

        # -----------------
        #  Train Generator
        # -----------------
        for _ in range(2):


            z = get_latent_samples((x.shape[0],z_dim))
            S_loss = -Loss(T, S, Q, z, x) 
            optS.zero_grad()
            S_loss.backward()
            optS.step()

    start_time = time.time()
    iterate_OTM(models['OTM'])
    timelist["OTM"] = time.time() - start_time

    #------------ RGM ------------
    def iterate_RGM(model):
        T      = model['T']
        S      = model['S']
        optT   = model['optT']
        optS   = model['optS']
        # ---------------------
        #  Train S
        # ---------------------
        for _ in range(2):
            optS.zero_grad()
            z = get_latent_samples((x.shape[0],z_dim))
            Tx   = T(x, feature=True)
            STx  = S(Tx)
            Sz   = S(z)
            TSz  = T(Sz, feature=True)
            S_loss = compute_RGM_cost(x,Tx,z,Sz) + ((x - STx)**2).sum()/x.shape[0] + ((z-TSz)**2).sum()/x.shape[0]
            cost_loss = S_loss
            cost_loss.backward()
            optS.step()

        # ---------------------
        #  Train T
        # ---------------------
        for _ in range(2):
            optT.zero_grad()
            z = get_latent_samples((x.shape[0],z_dim))
            Tx   = T(x, feature=True)
            STx  = S(Tx)
            Sz   = S(z)
            TSz  = T(Sz, feature=True)
            T_loss = compute_RGM_cost(x,Tx,z,Sz) + ((x - STx)**2).sum()/x.shape[0] + ((z-TSz)**2).sum()/x.shape[0]
            cost_loss = T_loss
            cost_loss.backward()
            optT.step()

    # start_time = time.time()
    # iterate_RGM(models['RGM'])
    # timelist["RGM"] = time.time() - start_time

    
    def iterate_VEEGAN():
        T      = models['VEEGAN']['T']
        S      = models['VEEGAN']['S']
        phi    = models['VEEGAN']['phi']
        optT   = models['VEEGAN']['optT']
        optS   = models['VEEGAN']['optS']
        optphi = models['VEEGAN']['optphi']




        valid = torch.ones((x.shape[0],  1), device=device).requires_grad_(False)
        fake  = torch.zeros((x.shape[0], 1), device=device).requires_grad_(False)

        # -----------------
        #  Train T (reconstructor)
        # -----------------

        optT.zero_grad()
        z = get_latent_samples((x.shape[0],z_dim))
        Sz = S(z)
        TS_loss = F.mse_loss(z, T.forward_T(Sz))
        TS_loss.backward()
        optT.step()

        # ---------------------
        #  Train S (generator)
        # ---------------------
        
        optS.zero_grad()
        z = get_latent_samples((x.shape[0],z_dim))
        Sz = S(z)
        TS_loss = F.mse_loss(z, T.forward_T(Sz))
        g_loss = phi(Sz, z).mean() + TS_loss
        g_loss.backward()
        optS.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optphi.zero_grad()
        z = get_latent_samples((x.shape[0],z_dim))
        fake_loss = adversarial_loss(sigmoid(phi(S(z), z)), valid)
        real_loss = adversarial_loss(sigmoid(phi(x, T.forward_T(x))), fake)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optphi.step()

    def iterate_VEEGAN2():
        T      = models['VEEGAN']['T']
        S      = models['VEEGAN']['S']
        phi    = models['VEEGAN']['phi']
        optT   = models['VEEGAN']['optT']
        optS   = models['VEEGAN']['optS']
        optphi = models['VEEGAN']['optphi']
        # -----------------
        #  Train Generator
        # -----------------



        valid = torch.ones((x.shape[0],  1),device=device).requires_grad_(False)
        fake  = torch.zeros((x.shape[0], 1),device=device).requires_grad_(False)

        optphi.zero_grad()
        z = get_latent_samples((x.shape[0],z_dim))
        gen_imgs = S(z)
        phi_loss = F.mse_loss(z,phi(gen_imgs, feature=True))
        phi_loss.backward()
        optphi.step()

        # ---------------------
        #  Train Reconstructor
        # ---------------------

        optS.zero_grad()
        z = get_latent_samples((x.shape[0],z_dim))
        gen_imgs = S(z)
        phi_loss = F.mse_loss(z,phi(gen_imgs, feature=True))
        g_loss = - adversarial_loss(T.gan(gen_imgs), fake) + phi_loss
        g_loss.backward()
        optS.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optT.zero_grad()
        real_loss = adversarial_loss(T.gan(x), valid)
        fake_loss = adversarial_loss(T.gan(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optT.step()

    
    start_time = time.time()
    iterate_VEEGAN()
    timelist['VEEGAN'] = time.time() - start_time


    #------------ VAEGAN ------------
    def iterate_VAEGAN():
        model = models['VAEGAN']
        T = model['T']
        S = model['S']
        psi = model['psi']
        optT = model['optT']
        optS = model['optS']  
        optpsi = model['optpsi']




        valid = torch.ones((x.shape[0], 1), device=device).requires_grad_(False)
        fake  = torch.zeros((x.shape[0], 1),device=device).requires_grad_(False)
        # -----------------
        #  Train T
        # -----------------
        optT.zero_grad()
        # Gaussian fitting
        mean, logvar = T.get_mean_and_var(x)
        # calculate KL loss
        KL_loss = - 0.5 * torch.sum(1+ logvar - mean**2 - logvar.exp())

        epsilon = torch.randn((logvar.shape[0], logvar.shape[1]), device=device)
        z = mean + logvar*epsilon
        x_hat = S(z)
        reproduction_loss = F.mse_loss(x_hat, x)
        loss = KL_loss + reproduction_loss
        loss.backward()
        optT.step()

        # -----------------
        #  Train S
        # -----------------
        optS.zero_grad()
        # Generate a batch of images
        mean, logvar = T.get_mean_and_var(x)

        epsilon = torch.randn((x.shape[0], z_dim),device=device).detach()
        z = mean + logvar * epsilon
        z = z.detach()
        x_hat = S(z)
        reproduction_loss = F.mse_loss(x_hat, x)
        # compute GAN loss
        z = get_latent_samples((x.shape[0],z_dim))
        fake_output = psi.gan(S(z))
        # loss_fake = F.binary_cross_entropy_with_logits(fake_output, valid)
        loss_fake = adversarial_loss(fake_output, valid)
        gamma = 5.0
        loss = gamma * reproduction_loss + loss_fake
        loss.backward()
        optS.step()

        optpsi.zero_grad()
        # Sample noise as generator input
        z = get_latent_samples((x.shape[0],z_dim))
        # Generate a batch of images
        Sz          = S(z).detach()
        real_output = psi.gan(x)
        fake_output = psi.gan(Sz)
        # loss_real = F.binary_cross_entropy_with_logits(real_output, valid)
        # loss_fake = F.binary_cross_entropy_with_logits(fake_output, fake)
        loss_real = adversarial_loss(real_output, valid)
        loss_fake = adversarial_loss(fake_output, fake)
        loss = loss_real + loss_fake
        loss.backward()
        optpsi.step()

        

    start_time = time.time()
    iterate_VAEGAN()
    timelist['VAEGAN'] = time.time() - start_time

    def iterate_GMEGAN(model,x,c_T=1,c_R=1,c_S=1,noise_scale=0.0,eta=lambda x:x):
        T      = model['T']
        S      = model['S']
        Rinv   = model['Rinv']
        psi    = model['psi']
        optG   = model['optG']
        optpsi = model['optpsi']

        def compute_psi_loss(psi, Sz, x):
            return psi(Sz).mean() - psi(x).mean() 
        
        p=6;k=2
        # ---------------------
        #  Train generator S
        # ---------------------
        optG.zero_grad()
        x_noise = x
        Tx   = T(x_noise, feature=True)
        T_loss = compute_GME_cost(x_noise, Tx, eta)

        z = get_latent_samples((x.shape[0],z_dim))
        Sz   = S(z)
        TSz  = T(Sz, feature=True)
        SRinvTx = S(Rinv(Tx));
        TSz  = T(Sz, feature=True);R_loss = F.mse_loss(z,TSz);S_loss = F.mse_loss(x,SRinvTx)
        psi_loss = compute_psi_loss(psi, Sz, x)
        loss = psi_loss +  R_loss * c_R + T_loss * c_T + S_loss * c_S
        loss.backward()
        optG.step()

        # ---------------------
        #  Train psi
        # ---------------------
        optpsi.zero_grad()
        z = get_latent_samples((x.shape[0],z_dim))
        Sz   = S(z)
        psi_loss = compute_psi_loss(psi, Sz, x)

        x_noise = x + torch.randn(x.shape, device=device) * noise_scale
        x_noise = x_noise.requires_grad_(True)
        real_validity = psi(x_noise)
        fake_validity = psi(Sz)

        real_grad_out  = torch.ones((x.shape[0], 1),device=device).requires_grad_(False)
        real_grad      = autograd.grad( real_validity, x_noise, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True )[0]
        real_grad_norm = real_grad.view(real_grad.shape[0], -1).pow(2).sum(1) ** (p/2) 


        fake_grad_out  = torch.ones((Sz.shape[0], 1),device=device).requires_grad_(False)
        fake_grad      = autograd.grad( fake_validity, Sz, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True )[0]
        fake_grad_norm = fake_grad.view(fake_grad.shape[0], -1).pow(2).sum(1) ** (p/2)

        k = 5
        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
        loss  = - psi_loss + div_gp
        
        loss.backward()
        optpsi.step()

        return T_loss

    start_time = time.time()
    T_loss= iterate_GMEGAN(models['GMEGAN'], x, c_T=10, c_R=0.5, c_S=5, noise_scale=0, eta=lambda x:torch.log(1+x))
    timelist["GMEGAN"] = time.time() - start_time

    # plotting
    if (it+1) % plot_every == 0:
        # get generator example
        with torch.no_grad():

            x = torch.tensor(real, device=device)

            ncols = 1 + len(titlelist)
            nrows = 3
            fig = plt.figure(figsize=(2.5*ncols,2.5*nrows),constrained_layout=True)
            gs = fig.add_gridspec(nrows=nrows, ncols=ncols)
            gs.update(wspace=0, hspace=0)

            ind = 0
            # ax = fig.add_subplot(2,ncols,ind)
            ax = fig.add_subplot(gs[0,ind])
            if cuda:
                x_cpu = x.cpu()
            else:
                x_cpu = x
            ax.scatter(x_cpu[:,0],x_cpu[:,1],c=label_val,alpha=0.5,cmap='tab20',vmin=0, vmax=max(centerind2label.values()))
            ax.set_title(f"$\\mu$ in {input_dim} D")
            ax.set_aspect('equal')
            
            ax = fig.add_subplot(gs[1,ind])
            if cuda:
                z_val_cpu = z_val.cpu()
            else:
                z_val_cpu = z_val
            ax.scatter(z_val_cpu[:,0],z_val_cpu[:,1],marker='x',s=z_val_sizes,c=z_val_labels,alpha=0.5,cmap='gist_rainbow')
            ax.set_title(f"$\\nu$ in {z_dim} D")
            ax.set_aspect('equal')
            
            for ind in range(1,ncols):
                title = titlelist[ind-1]
                T = models[title]['T']
                S = models[title]['S']
                subplot_model(ind, ax, gs, T, S, x, z_val, latent_val, label_val, z_val_sizes, z_val_labels, title, centers)

            filename = f'gaussian_comparison_{sigma}_{(it+1) // plot_every}.png'
            fig.savefig(os.path.join(save_fig_path, filename))
            plt.close('all')
            pbar.set_description(filename)
