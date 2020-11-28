import torch.nn as nn
import torch.nn.functional as F
import torch
from torchfusion.gan.applications import DCGANGenerator


class Generator(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()

        self.init_size = out_size[0] // 4
        in_channels = latent_size[0]
        self.feature_embedd = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2d(in_channels=in_channels, out_channels=20, kernel_size=5),
            nn.ReLU(),
            # state size. 20 x 10 x 10
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=3),
            nn.ReLU(),
            # state size. 16 x 8 x 8
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3),
            nn.ReLU(),
            # state size. 8 x 6 x 6
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=2),
            nn.ReLU(),
            # state size. (4) x 5 x 5
            nn.Flatten(),
            # nn.Linear(in_features=100, out_features=64),
            # nn.ReLU(),
        )
        self.generator = DCGANGenerator(latent_size=(100, 1, 1), output_size=out_size)

    def forward(self, z):
        features = self.feature_embedd(z)
        features = torch.reshape(features, shape=(-1, 100, 1, 1))
        return self.generator(features)
