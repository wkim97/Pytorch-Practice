from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#############################################################################################################
# Parameters settings
#############################################################################################################
# directory for dataset
dataroot = "./data/celeba"
workers = 2
batch_size = 128
# all training images will be transformed to this size
image_size = 64
# number of channels - 3 for color images
nc = 3
# size of z latent vector - size of generator input
nz = 100
# size of features maps in generator - length of the convolution layer
ngf = 64
# size of features maps in discriminator
ndf = 64
num_epochs = 50
lr = 0.0002
beta1 = 0.5
ngpu = torch.cuda.device_count()

#############################################################################################################
# Calling in dataset and plotting several images
#############################################################################################################
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),
                        (1, 2, 0)))
# plt.show()

#############################################################################################################
# Implementation - weights initialization
# weights_init takes a model as an input and re-initializes all layers
#############################################################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#############################################################################################################
# Implementation - generator
# Maps latent space vector z (noise) to data-space through deconvolution (ConvTranspose2d)
#############################################################################################################
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input latent vector z
            # ConvTranspose2d(channel_in, channel_out, kernel_size, stride, padding, bias)
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            # BatchNorm2d(num_features)
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # ngf x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # nc x 64 x 64
        )
    def forward(self, input):
        output = self.main(input)
        return output

#############################################################################################################
# Implementation - discriminator
# Takes an image as input and outputs a scalar probability of whether it is real or fake
# Series of Conv2d, BatchNorm2d, and LeakyReLU
#############################################################################################################
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is image of size nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

netG = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 2):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
print(netG)

netD = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 2):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
print(netD)

#############################################################################################################
# Loss functions and Optimizers
#############################################################################################################
criterion = nn.BCELoss()
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

#############################################################################################################
# Training discriminator and generator
#############################################################################################################
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        ########################################
        # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ########################################
        # Training with real image batch
        netD.zero_grad()
        real_cpu = data[0].to(device) # Real training image
        b_size = real_cpu.size(0) # size of batch
        label = torch.full((b_size,), real_label, device=device) # initial labels of batch_size size init. to 1
        output = netD(real_cpu).view(-1) # forward pass real batch through D
        errD_real = criterion(output, label) # calculate loss on all-real batch
        errD_real.backward() # calculate gradients for D in backward pass
        D_x = output.mean().item() # D(x), probability of x being real

        # Training with fake image batch
        noise = torch.randn(b_size, nz, 1, 1, device=device) # random noise of size batch_size by nz
        fake = netG(noise) # make fake images using random noise
        label.fill_(fake_label) # initial labels of batch_size size init. to 0
        output = netD(fake.detach()).view(-1) # forward pass fake batch through D
        errD_fake = criterion(output, label) # calculate loss on all-fake batch
        errD_fake.backward() # calculate gradients for D in backward pass
        D_G_z1 = output.mean().item() # D(G(z)) numerator
        errD = errD_real + errD_fake # Loss_D = discriminator loss
        optimizerD.step()

        ########################################
        # Update G network: maximize log(D(G(z))) - same as minimizing log(1-D(G(z)))
        ########################################
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1) # forward pass on all-fake batch through D
        errG = criterion(output, label) # calculate G's loss based on the output, loss_G = generator loss
        errG.backward() # calculate G's gradients
        D_G_z2 = output.mean().item() # D(G(z)) denominator
        optimizerG.step()

        ########################################
        # Training stats
        ########################################
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss-D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)):%.4f / %.4f'
                  % (epoch + 1, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1

print('Finished Training')
D_PATH = './discriminator.pth'
G_PATH = './generator.pth'
torch.save(netD.state_dict(), D_PATH)
torch.save(netG.state_dict(), G_PATH)

#############################################################################################################
# Plot loss graph
#############################################################################################################
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

#############################################################################################################
# Plot animation of fake images through training
#############################################################################################################
fig = plt.figure(figsize=(8, 8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

#############################################################################################################
# Plot real images and fake iamges side by side
#############################################################################################################
real_batch = next(iter(dataloader))

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),
                        (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()

