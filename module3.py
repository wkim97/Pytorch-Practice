import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

################################################
# Convolutional Neural Network
################################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel, 6 output channels, 3x3 conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 6 input channels, 16 output channels, 3x3 conv kernel
        self.conv2 = nn.Conv2d(6, 16, 3)
        # affine tranformation of 16 * 6 * 6 dim input to 120 dim output
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # pooling over window of 2x2 kernel after conv1
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # pooling after conv2
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # reshape the tensor
        x = x.view(-1, self.num_flat_features(x))
        # fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# parameters of models that are to be learned
params = list(net.parameters())
print(len(params))
print(params[0].size()) # [output channel, input channel, kernel size 1, kernel size 2]

input = torch.randn(1, 1, 32, 32) #nSamples x nChannels x Height x Width
out = net(input)
print(out) # Output of the CNN (size of the last FC layer)

net.zero_grad()
out.backward(torch.randn(1, 10))

################################################
# Loss Function
# looks at how far the output is from target
################################################
output = net(input)
target = torch.randn(10).view(1, -1)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss
print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # linear
print(loss.grad_fn.next_functions[0][0].next_functions[1][0]) # ReLu
print(loss.grad_fn.next_functions[0][0].next_functions[1][0].next_functions[0][0]) # linear
# 뭐 이런식으로 쭉 진행됨

################################################
# Backpropagation
# bias grad가 어떻게 바뀌는 지 보자!
################################################
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad) # [0, 0, 0, 0, 0, 0]

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad) # Same 1x6 array with conv1's bias gradients

################################################
# Updating Weights (Stochastic Gradient Descent)
# weight = weight - learning rate * gradient
################################################
learning_rate = 0.01
for f in net.parameters():
    # f.data = weight
    # f.grad.data = gradient
    # weight = weight - gradient * learning rate
    f.data.sub_(f.grad.data * learning_rate)

# SGD function in optim package
# Initialize optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.01)
# Learn in training loop
optimizer.zero_grad() # resets gradient buffer to 0 b/c otherwise, grad is stacked
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # does the gradient descent update