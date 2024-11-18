import torch
import torch.nn as nn
import numpy as np

class TransportT(torch.nn.Module):
    def __init__(self, input_shape = [3,32,32], z_dim = 100):
        super(TransportT, self).__init__()

        self.input_shape = input_shape
        self.z_dim = z_dim
        n_channels = input_shape[0]

        self.main1 = nn.Sequential(
            # Input is 3 x 32 x 32
            nn.Conv2d(n_channels, 32, (4, 4), (2, 2), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # State size. 16 x 16 x 16
            nn.Conv2d(32, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            # State size. 64 x 8 x 8
            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # State size. 128 x 4 x 4
        )
        self.main2 = nn.Sequential(
            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # State size. 256 x 2 x 2
        )
        self.output_main   = 256*2*2

        self.fc_T = nn.Sequential(
            nn.Conv2d(256, z_dim, kernel_size=2, stride=1, padding=0),
        )
        self.fc_phi = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=2, stride=1, padding=0),
        )

        self.output = nn.Sequential(
            nn.Tanh(),
        )
        
        self.sig = nn.Sequential(
            nn.Sigmoid(),
        )

        self.fc_VEEGAN = nn.Sequential(
            nn.Linear(z_dim*2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            )

        # latent mean and variance 
        self.mean_layer = nn.Sequential(nn.Linear(z_dim, z_dim))
        self.logvar_layer = nn.Sequential(nn.Linear(z_dim, z_dim))

    def get_mean_and_var(self, x):
        Tx = self.forward_T(x)
        return self.mean_layer(Tx), self.logvar_layer(Tx)

    def forward(self, x):
        x = x.view(x.shape[0], *self.input_shape)
        x = self.main2(self.main1(x))
        # x = x.view(x.shape[0], self.output_main)
        x = self.fc_phi(x).view(x.shape[0], 1)
        return x

    def forward_T(self, x):
        x = x.view(x.shape[0], *self.input_shape)
        x = self.main2(self.main1(x))
        # x = x.view(x.shape[0], self.output_main)
        x = self.fc_T(x).view(x.shape[0], self.z_dim)
        return x
    
    def forward_VEEGAN(self, x_input, z_input):
        x = self.forward_T(x_input)
        xz = torch.cat((x,z_input),1)
        return self.fc_VEEGAN(xz)

    def gw(self,x):
        x = self.forward(x)
        return self.output(x)

    def gan(self,x):
        x = self.forward(x)
        return self.sig(x)
    

class TransportG(torch.nn.Module):
    def __init__(self, z_dim = 100, output_shape = [1,32,32]):
        super(TransportG, self).__init__()

        self.z_dim = z_dim
        self.output_shape = output_shape

        self.init_channel = 32
        self.after_R_shape = (self.init_channel*4,8,8)
        

        self.R = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, np.prod(self.after_R_shape)),
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.init_channel * 4),
            nn.Upsample(scale_factor=2),
            # size: 1024 x 16 x 16
            nn.Conv2d(self.init_channel * 4, self.init_channel * 2, 5, stride=1, padding=2),
            nn.BatchNorm2d(self.init_channel * 2, 0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Upsample(scale_factor=2),
            # size: 512 x 32 x 32
            nn.Conv2d(self.init_channel * 2, self.init_channel * 1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.init_channel * 1, 0.8),
            nn.LeakyReLU(0.2,inplace=True),
            # size: 256 x 32 x 32
            nn.Conv2d(self.init_channel * 1, output_shape[0], 5, stride=1, padding=2),
            nn.Tanh(),
            # size: 3 x 32 x 32
        )

    def forward(self, x):
        x = self.R(x)
        x = x.view(x.shape[0], *self.after_R_shape)
        x = self.conv_blocks(x)
        return x

