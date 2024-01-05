import numpy as np
from jittor import nn


class Generator(nn.Module):

    def __init__(self, z_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers
        self.model = nn.Sequential(*block(self.z_dim, 128, normalize=False), *block(128, 256), *block(
            256, 512), *block(512, 1024), nn.Linear(1024, int(np.prod(self.img_shape))), nn.Tanh())

    def execute(self, z):
        img = self.model(z)
        img = img.view((img.shape[0], *self.img_shape))
        return img


class Discriminator(nn.Module):

    def __init__(self, z_dim=100, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.img_shape = img_shape
        self.model = nn.Sequential(nn.Linear(int(np.prod(self.img_shape)), 512), nn.LeakyReLU(
            scale=0.2), nn.Linear(512, 256), nn.LeakyReLU(scale=0.2), nn.Linear(256, 1))

    def execute(self, img):
        img_flat = img.view((img.shape[0], (- 1)))
        validity = self.model(img_flat)
        return validity
