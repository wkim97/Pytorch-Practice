import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##############################################################################
# Learning a Classifier
# 1. Call in image, text, or audio as NumPy array
# 2. Convert NumPy array into torch.*Tensor
# (Use Pillow or OpenCV for image)
# torchvision contains data loader for Imagenet, CIFAR10, MNIST, etc
# Use CIFAR10 for this classification exercise
# image size = 3 x 32 x 32 (3 channels of 32 x 32 pixels per image)
# 10 classification criteria
#############################################################################

#############################################################################
# 1. Call in CIFAR10 training and test data sets using torchvision,
# and normalize the data sets
#############################################################################
# Composes several transforms together
# transforms to tensor - convert PIL image or np.ndarray to tensor
# normalizes - normalize input to (input - mean) / std
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)
# 이미지용 데이터 변환기 data transformer
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=4, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=0)
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')
# Function to plot image
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
# Get random training image
dataiter = iter(trainloader)
images, labels = dataiter.next()
# Show image and label
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#############################################################################
# 2. Defining Convolutional Neural Network
# Fix CNN to take care of 3 channels per image
#############################################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input 3 channel, output 6 channel, 5 x 5 kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # pooling with kernel 2 x 2
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

#############################################################################
# 3. Defining loss functions and optimizer
# Use cross-entropy loss and stochastic gradient descent with momentum
#############################################################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#############################################################################
# 4. Learn the model by taking training data as input repeatedly
#############################################################################
for epoch in range(2): # repeat training multiple (2) times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # data = [inputs, labels]
        inputs, labels = data
        # change gradient to 0
        optimizer.zero_grad()
        # 순전파 + 역전파 + 최적화
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # prints out stats
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

