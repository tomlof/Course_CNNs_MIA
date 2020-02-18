#!/usr/bin/env python
# coding: utf-8

# # Assingment 1 - Malaria Cell Image Classification
# ### Course: Convolutional Neural Networks with Applications in Medical Image Analysis
# 
# Office hours: Mondays 13.15--15.00 (Tommy), Tuesdays 13.15--16.00 (Minh), Thursdays 08.15--12.00 (Attila)
# 
# Welcome. The first assignment is based on classifying images of cells, whether they are parasitized or uninfected by malaria. Your input will be an image of a cell, and your output is a binary classifier. It is based on an open dataset, available from Lister Hill National Center for Biomedical Communications (NIH): https://lhncbc.nlm.nih.gov/publication/pub9932. You need to download the file ftp://lhcftp.nlm.nih.gov/Open-Access-Datasets/Malaria/cell_images.zip (as described in the assignment instructions). The data was preprocessed and organized for easier machine learning applications.
# 
# Your task is to look through the highly customizable code below, which contains all the main steps for high accuracy classification of these data, and improve upon the model. The most important issues with the current code are noted in the comments for easier comprehension. Your tasks, to include in the report, are:
# 
# - Reach an accuracy of at least 96~\% on the validation dataset.
# - Plot the training/validating losses and accuracies. Describe when to stop training, and why that is a good choice.
# - Describe the thought process behind building your model and choosing the model hyper-parameters.
# - Describe what you think are the biggest issues with the current setup, and how to solve them.

# In[1]:


# Import necessary packages for loading the dataset

from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch import optim
from torch import nn
import torch
import matplotlib.pyplot as plt  # Package for plotting
import cv2
import os
import numpy as np  # Package for matrix operations, handling data
from torch.autograd import Variable
np.random.seed(2020)


# In[2]:


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


# In[3]:


# Path to dataset downloaded from the provided link
cwd = os.getcwd()
data_path = os.path.join(cwd, "Labs/data/cell_images")  # Path to dataset

# Check out dataset
parasitized_data = os.listdir(data_path + '/Parasitized/')
print(parasitized_data[:2])  # the output we get are the .png files
print("Number of parasitized images: " + str(len(parasitized_data)) + '\n')
uninfected_data = os.listdir(data_path + '/Uninfected/')
print(uninfected_data[:2])
print("Number of non-paratisized images: " + str(len(uninfected_data)))

# NOTE: The images are in .png format, they will have to be loaded individually and handled accordingly.


# In[4]:


# Look at some sample images
plt.figure(figsize=(12,5))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    img = plt.imread(data_path + '/Parasitized/' + parasitized_data[i])
    plt.imshow(img)
    plt.title('Image size: ' + str(np.shape(img)))
    plt.tight_layout()

plt.suptitle('Parasitized Image Samples')
# plt.show()

plt.figure(figsize=(12,4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    img = plt.imread(data_path + '/Uninfected/' + uninfected_data[i + 1])
    plt.imshow(img)
    plt.title('Image size: ' + str(np.shape(img)))
    plt.tight_layout()

plt.suptitle('Uninfected Image Samples')
# plt.show()

# NOTE: The images are of different size. Also they are RGB images.


# ### The dataset preprocessing so far has been to help you, you should not change anything. However, from now on, take nothing for granted.

# In[5]:


# Define transforms for the training/validation/testing data
transformation = transforms.Compose([transforms.Resize((16, 16)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])


# Split dataset into training and testing dataset
dataset = datasets.ImageFolder(root=data_path, transform=transformation)

# Find number of train/val/test images
num_images = len(dataset.targets)
test_size = 0.2
num_train = int((1-test_size)*num_images)
num_test = num_images - num_train
num_val = int(num_test/2)
num_test = num_test - num_val
print("Number of train: {} \t val: {} \t test: {}".format(
    num_train, num_val, num_test))
# NOTE: Keep the ratio of the split as it is. It will make evaluation easier for us.
# NOTE: The split should be reproducible, hence the random state.

train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, [num_train, num_val, num_test])


# In[6]:


# Train loader
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=batch_size,
                                           shuffle=True)

# Val loader
val_loader = torch.utils.data.DataLoader(val_set,
                                         batch_size=batch_size)

# Test loader
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=batch_size)


# In[7]:


class CellModel(nn.Module):
    def __init__(self):
        super(CellModel, self).__init__()
        # input is 16x16
        # padding=1 for same padding
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        # feature map size is 8x8 by pooling
        # padding=1 for same padding
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        # feature map size is 4x4 by pooling
        self.fc1 = nn.Linear(8*4*4, 16)
        self.fc2 = nn.Linear(16, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 8*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# NOTE: Are the input sizes correct?
# NOTE: Are the output sizes correct?
# NOTE: Is the 'hotmap' activation layer in the model?
# NOTE: Try to imagine the model layer-by-layer and think it through. Is it doing something reasonable?
# NOTE: Are the model parameters split "evenly" between the layers? Or is there one huge layer?
# NOTE: Will the model fit into memory? Is the model too small? Is the model too large?   


# NOTE: Are you satisfied with the loss function?
# NOTE: Are you satisfied with the metric?
# NOTE: Are you satisfied with the optimizer and its parameters?

    
model = CellModel()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# check model parameter
for p in model.parameters():
    print(p.size())


# ## Train

# In[8]:


train_loss = []
train_accu = []
i = 0
for epoch in range(3):
    model.train()
    i = 0
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
        accuracy = prediction.eq(target.data).sum().item()/batch_size*100
        train_accu.append(accuracy)
        if i % 100 == 0:
            print('Epoch: {}\t Step: {} \t\t Loss: {:.3f} \t Accuracy: {:.3f}'.format(
                epoch, i, loss.item(), accuracy))
        i += 1

    with torch.no_grad():
        correct = 0
        for data, target in val_loader:
            data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

        print('Validation accuracy: {:.2f}%'.format(
            100. * correct / len(val_loader.dataset)))


# ## Test

# In[ ]:


with torch.no_grad():
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        output = model(data)
        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()

    print('Testing accuracy: {:.2f}%'.format(
        100. * correct / len(test_loader.dataset)))

