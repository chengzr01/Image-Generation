import jittor.transform as transform
from jittor.dataset.mnist import MNIST
import argparse
import os
import numpy as np
import time
import cv2

import jittor as jt
from jittor import init
from jittor import nn
jt.flags.use_cuda = 1

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="adam: learning rate")
parser.add_argument("--beta", type=float, default=4,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10,
                    help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int,
                    default=3000, help="interval between image sampling")
opt = parser.parse_args()
print(opt)


def kl_divergence(mu, logvar):
    klds = jt.array(- 0.5 * (1 + logvar - mu.pow(2) - jt.exp(logvar)))
    total_kld = jt.float32(klds.sum(1).mean(0, True))
    return total_kld


def reparameterization(mu, logvar):
    std = jt.exp(logvar / 2)
    sampled_z = jt.array(np.random.normal(
        0, 1, (mu.shape[0], opt.latent_dim))).float32()
    z = sampled_z * std + mu
    return z


img_shape = (opt.channels, opt.img_size, opt.img_size)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.Leaky_relu(
                0.2), nn.Linear(512, 512),
            nn.Leaky_relu(0.2)
        )
        self.mu = nn.Linear(512, opt.latent_dim)
        self.logvar = nn.Linear(512, opt.latent_dim)

    def execute(self, img):
        img_flat = jt.reshape(img, [img.shape[0], (- 1)])
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.Leaky_relu(
                0.2),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def execute(self, z):
        img_flat = self.model(z)
        img = jt.reshape(img_flat, [img_flat.shape[0], *img_shape])
        return img


# Use binary cross-entropy loss
reconstruction_loss = nn.MSELoss(reduction='sum')

# Initialize generator and discriminator
encoder = Encoder()
decoder = Decoder()

# Configure data loader

transform = transform.Compose([
    transform.Resize(opt.img_size),
    transform.Gray(),
    transform.ImageNormalize(mean=[0.5], std=[0.5]),
])
train_loader = MNIST(train=True, transform=transform).set_attrs(
    batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = nn.Adam(
    encoder.parameters() + decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
)


def save_image(img, path, nrow=10, padding=5):
    N, C, W, H = img.shape
    if (N % nrow != 0):
        print("N%nrow!=0")
        return
    ncol = int(N/nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
            img_.append(np.zeros((C, W, padding)))
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C, padding, img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C, padding, img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C, img.shape[1], padding)), img], 2)
    min_ = img.min()
    max_ = img.max()
    img = (img-min_)/(max_-min_)*255
    img = img.transpose((1, 2, 0))
    if C == 3:
        img = img[:, :, ::-1]
    cv2.imwrite(path, img)


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    print("[DEBUG] sample image")
    z = jt.array(np.random.normal(
        0, 1, (n_row ** 2, opt.latent_dim))).float32().stop_grad()
    gen_imgs = decoder(z)
    beta = opt.beta
    save_image(gen_imgs.numpy(),
               f"./images/{beta}_{batches_done}.png", nrow=n_row)


for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(train_loader):
        sta = time.time()
        real_imgs = jt.array(imgs).stop_grad()

        encoded_imgs, mu, logvar = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        g_loss = reconstruction_loss(
            real_imgs, decoded_imgs).div(opt.batch_size) + opt.beta * kl_divergence(mu, logvar)

        optimizer_G.step(g_loss)

        jt.sync_all()
        if i % 50 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [Time: %f]"
                % (epoch, opt.n_epochs, i, len(train_loader), g_loss.data[0], time.time() - sta)
            )

        batches_done = epoch * len(train_loader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
