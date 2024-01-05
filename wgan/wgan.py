import jittor as jt
from jittor import init
from jittor import nn
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
import argparse
import os
import numpy as np
import math
import sys
import cv2
import time

from model import Generator, Discriminator
from utils import save_image

jt.flags.use_cuda = 1
os.makedirs('images', exist_ok=True)


def main(opt):
    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # Model
    generator = Generator()
    discriminator = Discriminator()
    dataloader = MNIST(train=True, transform=transform.Compose([
        transform.Resize(size=opt.img_size),
        transform.Gray(),
        transform.ImageNormalize(mean=[0.5], std=[0.5]),
    ])).set_attrs(
        batch_size=opt.batch_size, shuffle=True)

    # Optimizers
    optimizer_G = jt.optim.RMSprop(generator.parameters(), lr=opt.lr)
    optimizer_D = jt.optim.RMSprop(discriminator.parameters(), lr=opt.lr)
    batches_done = 0

    warmup_times = -1
    run_times = 3000
    total_time = 0.
    cnt = 0

    #  Training
    for epoch in range(opt.n_epochs):
        for (i, (real_imgs, _)) in enumerate(dataloader):

            #  Train Discriminator
            z = jt.array(np.random.normal(
                0, 1, (real_imgs.shape[0], opt.latent_dim)).astype(np.float32))
            fake_imgs = generator(z).detach()
            loss_D = ((- jt.mean(discriminator(real_imgs))) +
                      jt.mean(discriminator(fake_imgs)))
            optimizer_D.step(loss_D)
            for p in discriminator.parameters():
                p.assign(p.maximum(- opt.clip_value).minimum(opt.clip_value))

            #  Train Generator
            if ((i % opt.n_critic) == 0):
                gen_imgs = generator(z)
                loss_G = (- jt.mean(discriminator(gen_imgs)))
                optimizer_G.step(loss_G)
                if warmup_times == -1:
                    print(('[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' % (epoch, opt.n_epochs,
                                                                                      (batches_done % len(dataloader)), len(dataloader), loss_D.numpy()[0], loss_G.numpy()[0])))

            if warmup_times == -1:
                if ((batches_done % opt.sample_interval) == 0):
                    save_image(
                        gen_imgs.data[:25], ('images/%d.png' % batches_done), nrow=5)
                batches_done += 1
            else:
                jt.sync_all()
                cnt += 1
                print(cnt)
                if cnt == warmup_times:
                    jt.sync_all(True)
                    sta = time.time()
                if cnt > warmup_times + run_times:
                    jt.sync_all(True)
                    total_time = time.time() - sta
                    print(
                        f"run {run_times} iters cost {total_time} seconds, and avg {total_time / run_times} one iter.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='size of the batches')
    parser.add_argument('--lr', type=float, default=5e-05,
                        help='learning rate')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--img_size', type=int, default=28,
                        help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1,
                        help='number of image channels')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='number of training steps for discriminator per iter')
    parser.add_argument('--clip_value', type=float, default=0.01,
                        help='lower and upper clip value for disc. weights')
    parser.add_argument('--sample_interval', type=int,
                        default=3000, help='interval betwen image samples')
    opt = parser.parse_args()
    main(opt)
