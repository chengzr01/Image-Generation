import argparse
import os
import numpy as np
import time
import cv2

import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
from jittor.dataset.mnist import MNIST

from model import Encoder, Decoder
from utils import kl_divergence, save_image

jt.flags.use_cuda = 1
os.makedirs("images", exist_ok=True)


def main(opt):
    img_shape = (opt.channels, opt.img_size, opt.img_size)
    # Model
    encoder = Encoder()
    decoder = Decoder()

    train_loader = MNIST(train=True, transform=transform.Compose([
        transform.Resize(opt.img_size),
        transform.Gray(),
        transform.ImageNormalize(mean=[0.5], std=[0.5]),
    ])).set_attrs(
        batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = nn.Adam(
        encoder.parameters() + decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    # Loss
    reconstruction_loss = nn.MSELoss(reduction='sum')

    # Training
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
                z = jt.array(np.random.normal(
                    0, 1, (10 ** 2, 10))).float32().stop_grad()
                gen_imgs = decoder(z)
                beta = opt.beta
                save_image(gen_imgs.numpy(),
                           f"./images/{beta}_{batches_done}.png", nrow=10)


if __name__ == "__main__":
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
    main(opt)
