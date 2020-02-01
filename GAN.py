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

num_epochs = 100
train_hist = {}

#############################################################################################################
# Call in data
#############################################################################################################
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

datasets = dset.MNIST(root='./data/MNIST', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(mnist, datasets, batch_size=60, shuffle=True)

use_gpu = False
if torch.cuda.is_available():
    use_gpu = True
leave_log = True
if leave_log:
    result_dir = './results/GAN_generated_images'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

#############################################################################################################
# Generator
#############################################################################################################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=100, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=28 * 28),
            nn.Tanh())

    def forward(self, inputs):
        return self.main(inputs).view(-1, 1, 28, 28)

#############################################################################################################
# Discriminator
#############################################################################################################
class Discriminator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid())

    def forward(self, inputs):
        inputs = inputs.view(-1, 28 * 28)
        return self.main(inputs)

G = Generator()
D = Discriminator()
if use_gpu:
    G.cuda()
    D.cuda()

#############################################################################################################
# Criterion and optimizer
#############################################################################################################
criterion = nn.BCELoss()
G_optimizer = optim.Adam(G.parameters((), lr=0.0002, betas=(0.5, 0.999)))
D_optimizer = optim.Adam(D.parameters((), lr=0.0002, betas=(0.5, 0.999)))

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

z_fixed = Variable(torch.randn(5 * 5, 100), volatile=True)
if use_gpu:
    z_fixed = z_fixed.cuda()

for epoch in range(num_epochs):
    if leave_log:
        D_losses = []
        G_losses = []

    for real_data, _ in dataloader:
        batch_size = real_data.size(0)
        real_data = Variable(real_data)
        label_real = Variable(torch.ones(batch_size, 1))
        label_fake = Variable(torch.zeros(batch_size, 1))
        if use_gpu:
            real_data, label_real, label_fake = real_data.cuda(), label_real.cuda(), label_fake.cuda()
        D_x = D(real_data)
        D_loss_real = criterion(D_x, label_real)

        z = Variable(torch.randn((batch_size, 100)))
        if use_gpu:
            z = z.cuda()
        fake_data = G(z)
        D_G_z = D(fake_data)
        D_loss_fake = criterion(D_G_z, label_fake)
        D_loss = D_loss_real + D_loss_fake

        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        if leave_log:
            D_losses.append(D_loss.data[0])

        z = Variable(torch.randn((batch_size, 100)))
        if use_gpu:
            z = z.cuda()
        fake_data = G(z)
        D_G_z = D(fake_data)
        G_loss = criterion(D_G_z, label_fake)

        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        if leave_log:
            G_lossses.append(G_loss.data[0])

    if leave_log:
        true_positive_rate = (D_x > 0.5).float().mean().data[0] # Probability real image classified as real
        true_negative_rate = (D_G_z < 0.5).float().mean().data[0] # Probability fake image classified as fake
        base_message = ("Epoch: {epoch:<3d} D_Loss: {d_loss:<8.6} G_Loss: {g_loss:<8.6} "
                        "True Positive Rate: {tpr:<5.1%} True Negative Rate: {tnr:<5.1%")
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
        square_plot(fake_data_fixed.view(25, 28, 28).cpu().data.numpy(), path=image_path)
        generated_images.append(image_path)
    if leave_log:
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

torch.save(G.state_dict(), "./models/gan_generator.pkl")
torch.save(D.state_dict(), "./models/gan_discriminator.pkl")
with open('./models/gan_train_history.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
generated_image_array = [imageio.imread(generated_image) for generated_image in generated_images]
imageio.mimsave(result_dir + '/GAN_generation.gif', generated_image_array, fps=5)