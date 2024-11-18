import torch
import torch.nn as nn

class TransportT(torch.nn.Module):
    def __init__(self, input_shape = [3,32,32], z_dim = 100):
        super(TransportT, self).__init__()

        self.out_dim = 8*8*8
        self.out = nn.Sequential(
            nn.Linear(self.out_dim,z_dim)
        )

        self.input_shape = input_shape
        n_channels = input_shape[0]

        img_size = input_shape[1]

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
        self.output_main_T = 128*(img_size/8)**2
        self.output_main   = 256*2*2

        self.fc_T = nn.Sequential(
            nn.Linear(self.output_main, z_dim),
        )
        # self.fc_phi = nn.Sequential(
        #     nn.Linear(self.output_main, 1),
        # )

        self.fc_phi = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
        )

        self.output = nn.Sequential(
            nn.Tanh(),
        )
        
        self.sig = nn.Sequential(
            nn.Sigmoid(),
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
        x = x.view(x.shape[0], self.output_main)
        x = self.fc_phi(self.fc_T(x))
        return x

    def forward_T(self, x):
        x = x.view(x.shape[0], *self.input_shape)
        x = self.main2(self.main1(x))
        x = x.view(x.shape[0], self.output_main)
        x = self.fc_T(x)
        return x

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

        self.R = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, z_dim),
        )

        self.tmp = nn.Sequential(
            nn.Linear(z_dim, 256*4*4),
        )

        self.S = nn.Sequential(
            # Input is 100, going into a convolution.
            # nn.ConvTranspose2d(z_dim, 256, (4, 4), (1, 1), (0, 0)),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 4 x 4
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 8 x 8
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 16 x 16
            nn.ConvTranspose2d(64, output_shape[0], (4, 4), (2, 2), (1, 1)),
            # state size. 3 x 32 x 32
            nn.Tanh()
        )

    def forward(self, x):
        x = self.tmp(self.R(x))
        x = x.view(x.shape[0], 256, 4, 4)
        x = self.S(x)
        return x

    def forward_S(self,x):
        x = self.tmp(x)
        x = x.view(x.shape[0], 256, 4, 4)
        x = self.S(x)
        return x

