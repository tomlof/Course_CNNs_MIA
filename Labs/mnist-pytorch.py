#!/usr/bin/env python
# coding: utf-8

# # Exercises 1 - MNIST Optical Character Recognition
# ### Course: Convolutional Neural Networks with Applications in Medical Image Analysis
# Office hours: Mondays 13.15--15.00 (Tommy), Tuesdays 13.15--16.00 (Minh), Thursdays 08.15--12.00 (Attila)
# 
# Below is an example notebook for a simple Keras pipeline. The dataset is MNIST (http://yann.lecun.com/exdb/mnist/), where each image is of a handwritten digit of 0-9.

# ## Initialize

# In[ ]:


import matplotlib.pyplot as plt 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# In[ ]:


SEED = 1

# See if GPU is available to use
cuda = torch.cuda.is_available()

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    use_cuda = True
else:
    use_cuda = False
    
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


# ## Dataset

# In[ ]:


batch_size = 50

# Train loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)


# In[ ]:


# Test loader
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=1000)


# ## Model

# In[ ]:


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        # input is 28x28
        # padding=2 for same padding
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        # feature map size is 14*14 by pooling
        # padding=1 for same padding
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # feature map size is 7*7 by pooling
        self.fc1 = nn.Linear(8*7*7, 16)
        self.fc2 = nn.Linear(16, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 8*7*7)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
model = MnistModel()
model = model.to(device)
model


# In[ ]:


# check model parameter
for p in model.parameters():
    print(p.size())


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=0.0001)


# ## Train

# In[ ]:


model.train()
train_loss = []
train_accu = []
i = 0
for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()    # calc gradients
        train_loss.append(loss.item())
        optimizer.step()   # update gradients
        prediction = output.data.max(1)[1]   # first column has actual prob.
        accuracy = prediction.eq(target.data).sum()/batch_size*100
        train_accu.append(accuracy)        
        if i % 100 == 0:
            print('Epoch: {}\tTrain Step: {}\t\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(epoch, i, loss.item(), accuracy))
        i += 1


# In[ ]:


plt.plot(np.arange(len(train_loss)), train_loss)


# In[ ]:


plt.plot(np.arange(len(train_accu)), train_accu)


# ## Evaluate

# In[ ]:


model.eval()
correct = 0
for data, target in test_loader:
    data, target = Variable(data, volatile=True), Variable(target)
    data, target = data.to(device), target.to(device)
    data, target = Variable(data), Variable(target)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).sum()

print('\nTest set: Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))

