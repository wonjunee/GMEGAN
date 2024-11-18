import torch
import torch.nn as nn
import numpy as np

class TransportT(torch.nn.Module):
    def __init__(self, input_shape = [3,32,32], z_dim = 100):
        super(TransportT, self).__init__()

        self.input_shape = input_shape
        self.z_dim = z_dim
        n_channels = input_shape[0]

        conv_channel = 16

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2,True), 
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.main = nn.Sequential(
            *discriminator_block(n_channels, conv_channel, bn=False),
            *discriminator_block(conv_channel, conv_channel*2),
            *discriminator_block(conv_channel*2, conv_channel*4),
        )

        self.output_main   = conv_channel*4*4*4

        self.fc_T = nn.Sequential(
            nn.Conv2d(conv_channel*4, z_dim, kernel_size=4, stride=1, padding=0),
        )

        # self.conv_T = nn.Sequential(
        #     nn.Conv2d(conv_channel*4, z_dim, kernel_size=4, stride=1, padding=0),
        # )

        self.fc_phi = nn.Sequential(
            nn.Conv2d(conv_channel*4, 1, kernel_size=4, stride=1, padding=0),
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

        self.fc_VEEGAN = nn.Sequential(
            nn.Linear(z_dim*2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            )

    def get_mean_and_var(self, x):
        Tx = self.forward_T(x)
        return self.mean_layer(Tx), self.logvar_layer(Tx)

    def forward(self, x):
        x = x.view(x.shape[0], *self.input_shape)
        x = self.main(x)
        x = self.fc_phi(x).view(x.shape[0],1)
        return x
    

    def forward_T(self, x):
        x = x.view(x.shape[0], *self.input_shape)
        x = self.main(x)
        # x = x.view(x.shape[0], self.output_main)
        x = self.fc_T(x).view(x.shape[0],self.z_dim)
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
    
    def ugan(self, backup):
        for m_from, m_to in zip(backup.modules(), self.modules()):
            if isinstance(m_to, nn.Module):
                continue  # Skip non-linear layers
            if isinstance(m_to, nn.Parameter):
                m_to.data = m_from.data.clone()
 

class TransportG(torch.nn.Module):
    def __init__(self, z_dim = 100, output_shape = [1,32,32]):
        super(TransportG, self).__init__()

        self.z_dim = z_dim
        self.output_shape = output_shape

        self.init_channel = 16
        self.init_size = 32 // 4
        self.init_shape = (self.init_channel * 4, self.init_size, self.init_size)

        self.l1 = nn.Sequential(nn.Linear(z_dim, np.prod(self.init_shape)))
        # size: 1024 x 8 x 8

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.init_channel * 4),
            nn.Upsample(scale_factor=2),
            # size: 1024 x 16 x 16
            nn.Conv2d(self.init_channel * 4, self.init_channel * 2, 5, stride=1, padding=2),
            nn.BatchNorm2d(self.init_channel * 2, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            # size: 512 x 32 x 32
            nn.Conv2d(self.init_channel * 2, self.init_channel * 1, 5, stride=1, padding=2),
            nn.BatchNorm2d(self.init_channel * 1, 0.8),
            nn.ReLU(inplace=True),
            # size: 256 x 32 x 32
            nn.Conv2d(self.init_channel * 1, output_shape[0], 5, stride=1, padding=2),
            nn.Tanh(),
            # size: 3 x 32 x 32
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], *self.init_shape)
        img = self.conv_blocks(out)
        return img

