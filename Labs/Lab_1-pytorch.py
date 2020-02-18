
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
np.random.seed(2020)


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


# Path to dataset downloaded from the provided link
data_path = "/home/minhvu/github/Course_CNNs_MIA/Labs/data/cell_images"  # Path to dataset


# Check out dataset
parasitized_data = os.listdir(data_path + '/Parasitized/')
print(parasitized_data[:2])  # the output we get are the .png files
print("Number of parasitized images: " + str(len(parasitized_data)) + '\n')
uninfected_data = os.listdir(data_path + '/Uninfected/')
print(uninfected_data[:2])
print("Number of non-paratisized images: " + str(len(uninfected_data)))

# NOTE: The images are in .png format, they will have to be loaded individually and handled accordingly.


# Look at some sample images
plt.figure(figsize=(12, 5))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    img = plt.imread(data_path + '/Parasitized/' + parasitized_data[i])
    plt.imshow(img)
    plt.title('Image size: ' + str(np.shape(img)))
    plt.tight_layout()

plt.suptitle('Parasitized Image Samples')
# plt.show()

plt.figure(figsize=(12, 4))
for i in range(4):
    plt.subplot(1, 4, i + 1)
    img = plt.imread(data_path + '/Uninfected/' + uninfected_data[i + 1])
    plt.imshow(img)
    plt.title('Image size: ' + str(np.shape(img)))
    plt.tight_layout()

plt.suptitle('Uninfected Image Samples')
# plt.show()


# Define transforms for the training/validation/testing data
train_transforms = transforms.Compose([transforms.Resize(16),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

val_transforms = transforms.Compose([transforms.Resize(16),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(16),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

transforms = transforms.Compose([transforms.Resize(16),
                                 transforms.ToTensor()])


# Split dataset into training and testing dataset
dataset = datasets.ImageFolder(root=data_path, transform=transforms)

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


# Train loader
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=32,
                                           shuffle=True)

# Val loader
val_loader = torch.utils.data.DataLoader(val_set,
                                         batch_size=32)

# Test loader
test_loader = torch.utils.data.DataLoader(test_set,
                                          batch_size=32)


class CellModel(nn.Module):
    def __init__(self):
        super(CellModel, self).__init__()
        # input is 16x16
        # padding=1 for same padding
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
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


model = CellModel()
model = model.to(device)

# check model parameter
for p in model.parameters():
    print(p.size())

optimizer = optim.Adam(model.parameters(), lr=0.0001)

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
            print('Epoch: {}\tTrain Step: {}\t\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(
                epoch, i, loss.item(), accuracy))
        i += 1
