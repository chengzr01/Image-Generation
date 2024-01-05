import numpy as np
import jittor as jt
from jittor import nn


class Encoder(nn.Module):
    def __init__(self, z_dim=10, img_shape=(1, 32, 32)):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.Leaky_relu(
                0.2), nn.Linear(512, 512),
            nn.Leaky_relu(0.2)
        )
        self.mu = nn.Linear(512, self.z_dim)
        self.logvar = nn.Linear(512, self.z_dim)

    def execute(self, img):
        img_flat = jt.reshape(img, [img.shape[0], (- 1)])
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        std = jt.exp(logvar / 2)
        sampled_z = jt.array(np.random.normal(
            0, 1, (mu.shape[0], self.z_dim))).float32()
        z = sampled_z * std + mu
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim=10, img_shape=(1, 32, 32)):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.Leaky_relu(
                0.2),
            nn.Linear(512, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def execute(self, z):
        img_flat = self.model(z)
        img = jt.reshape(img_flat, [img_flat.shape[0], *self.img_shape])
        return img
