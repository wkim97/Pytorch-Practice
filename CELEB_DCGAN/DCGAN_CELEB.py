from __future__ import print_function
import argparse
import os
import random
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch.autograd import Variable
from torchvision.utils import save_image

num_epochs = 50
train_hist = {}
b_size = 64
ngpu = torch.cuda.device_count()
workers = 2
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
lr = 0.0002
beta1 = 0.5
dataroot = './data/celeba'

#############################################################################################################
# Call in data
#############################################################################################################
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=b_size, shuffle=True, num_workers=workers)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

use_gpu = False
if torch.cuda.is_available():
    use_gpu = True
leave_log = True
if leave_log:
    result_dir = './results/group_outputs'
    save_dir = './results/individual_outputs'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

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
# Generator
#############################################################################################################
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8,
                               kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_features=ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.Tanh())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
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
            nn.Conv2d(in_channels=nc, out_channels=ndf,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=ndf * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0,
                      bias=False),
            nn.Sigmoid())

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)

G = Generator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 2):
    G = nn.DataParallel(G, list(range(ngpu)))
G.apply(weights_init)
print(G)

D = Discriminator(ngpu).to(device)
if (device.type == 'cuda') and (ngpu > 2):
    D = nn.DataParallel(D, list(range(ngpu)))
D.apply(weights_init)
print(D)

if use_gpu:
    G.cuda()
    D.cuda()

#############################################################################################################
# Criterion and optimizer
#############################################################################################################
criterion = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

#############################################################################################################
# Visualizing results
#############################################################################################################
def square_plot(data, path):
    if type(data) == list:
        data = np.concatentate(data)
    data = (data - data.min()) / (data.max() - data.min())
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1)) + ((0, 0),) * (data.ndim - 3))
    data = np.pad(data, padding, mode='constant', constant_values=1)
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imsave(path, data, cmap='gray')

#############################################################################################################
# Training
#############################################################################################################
if leave_log:
    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    generated_images = []

z_fixed = Variable(torch.randn(b_size, 100, 1, 1))
if use_gpu:
    z_fixed = z_fixed.cuda()

img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(num_epochs):
    if leave_log:
        D_losses = []
        G_losses = []
    for i, (real_data, _) in enumerate(dataloader):
        batch_size = real_data.size(0)
        real_data = Variable(real_data)
        label_real = Variable(torch.ones(batch_size))
        label_fake = Variable(torch.zeros(batch_size))
        if use_gpu:
            real_data, label_real, label_fake = real_data.cuda(), label_real.cuda(), label_fake.cuda()
        z = Variable(torch.randn((batch_size, 100, 1, 1)))
        if use_gpu:
            z = z.cuda()
        fake_data = G(z)

        D.zero_grad()
        real_output = D(real_data)
        D_loss_real = criterion(real_output, label_real)

        fake_output = D(fake_data)
        D_loss_fake = criterion(fake_output, label_fake)
        D_loss = D_loss_real + D_loss_fake
        D_loss.backward()
        D_optimizer.step()
        D_x = real_output.mean().item()

        if leave_log:
            D_losses.append(D_loss.data)

        z = Variable(torch.randn((batch_size, 100, 1, 1)))
        if use_gpu:
            z = z.cuda()
        fake_data = G(z)
        fake_output = D(fake_data)
        G.zero_grad()
        G_loss = criterion(fake_output, label_real)

        G_loss.backward()
        G_optimizer.step()
        D_G_z = fake_output.mean().item()
        if leave_log:
            G_losses.append(G_loss.data)

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)):%.4f'
                  % (epoch + 1, num_epochs, i, len(dataloader),
                     D_loss.item(), G_loss.item(), D_x, D_G_z))
            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = G(z_fixed).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            iters += 1

    # Save individual images at the end of training
    if epoch == num_epochs - 1:
        for j in range(500):
            path = save_dir + '/image{}.png'.format(j)
            noise = Variable(torch.randn(1, 100, 1, 1)).cuda()
            image = F.interpolate(G(noise).cpu(), size=32).view(3, 32, 32)
            vutils.save_image(image, path, normalize=True)

    if leave_log:
        true_positive_rate = (real_output > 0.5).float().mean().data # Probability real image classified as real
        true_negative_rate = (fake_output < 0.5).float().mean().data # Probability fake image classified as fake
        base_message = ("Epoch: {epoch:<3d} D_Loss: {d_loss:<8.6} G_Loss: {g_loss:<8.6} "
                        "True Positive Rate: {tpr:<5.1%} True Negative Rate: {tnr:<5.1%}")
        message = base_message.format(
            epoch=epoch,
            d_loss=sum(D_losses)/len(D_losses),
            g_loss=sum(G_losses)/len(G_losses),
            tpr=true_positive_rate,
            tnr=true_negative_rate)
        print(message)
    if leave_log:
        fake_data_fixed = G(z_fixed)
        image_path = result_dir + '/epoch{}.png'.format(epoch)

        fig = plt.figure()
        plt.axis("off")
        sample = np.transpose(vutils.make_grid(fake_data_fixed.view(b_size, 3, 64, 64),
                                nrow=8, padding=3, normalize=True).cpu().detach().numpy(), (1, 2, 0))
        plt.imsave(image_path, sample)
        plt.close(fig)
        generated_images.append(image_path)
    if leave_log:
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

torch.save(G.state_dict(), './models/dcgan_generator.pkl')
torch.save(D.state_dict(), './models/dcgan_discriminator.pkl')

generated_image_array = [imageio.imread(generated_image) for generated_image in generated_images]
imageio.mimsave(result_dir + '/DCGAN_generation.gif', generated_image_array, fps=5)

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
