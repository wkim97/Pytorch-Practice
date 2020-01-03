import torch
import torchvision
import torchvision.transforms as transforms

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
    trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')
