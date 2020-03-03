from torch.utils.data import Dataset, DataLoader
import torch
import torch.utils.data
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt


class BratsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 inputs,
                 outputs
                 ):
        self.data_path = data_path
        self.inputs = inputs
        self.outputs = outputs
        if data_path is None:
            raise ValueError('The data path is not defined.')
        if not os.path.isdir(data_path):
            raise ValueError('The data path is incorrectly defined.')
        self.file_idx = 0
        self.file_list = [self.data_path + '/' + s for s in
                          os.listdir(self.data_path)]
        self.on_epoch_end()
        with np.load(self.file_list[0]) as npzfile:
            self.out_dims = []
            self.in_dims = []
            self.n_channels = 1
            for i in range(len(self.inputs)):
                im = npzfile[self.inputs[i]]
                self.in_dims.append((self.n_channels,
                                     *np.shape(im)))
            for i in range(len(self.outputs)):
                im = npzfile[self.outputs[i]]
                self.out_dims.append((self.n_channels,
                                      *np.shape(im)))
            npzfile.close()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.file_list))))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index:(index + 1)]
        print(indexes)
        # Find list of IDs
        list_IDs_temp = [self.file_list[k] for k in indexes]
        # Generate data
        i, o = self.__data_generation(list_IDs_temp)
        return i, o

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))

    def __data_generation(self, temp_list):
        'Generates data containing samples'
        # Initialization
        inputs = []
        outputs = []
        for i in range(self.inputs.__len__()):
            inputs.append(np.empty(self.in_dims[i]).astype(np.single))
        for i in range(self.outputs.__len__()):
            outputs.append(np.empty(self.out_dims[i]).astype(np.single))
        for i, ID in enumerate(temp_list):
            with np.load(ID) as npzfile:
                for idx in range(len(self.inputs)):
                    x = npzfile[self.inputs[idx]] \
                        .astype(np.single)
                    x = np.expand_dims(x, axis=0)
                    inputs[idx][i, ] = x
                for idx in range(len(self.outputs)):
                    x = npzfile[self.outputs[idx]] \
                        .astype(np.single)
                    x = np.expand_dims(x, axis=0)
                    outputs[idx][i, ] = x
                npzfile.close()
        return inputs, outputs


gen_dir = "C:/Users/minhm/Documents/GitHub/Course_CNNs_MIA/projects/data/"
dataset = BratsDataset(data_path=gen_dir + 'training',
                       inputs=['flair', 't1'],
                       outputs=['mask'])

dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=0)


for img_in, img_out in dataloader:
    plt.subplot(1, 3, 1)
    plt.imshow(img_in[0][0, 0, :, :], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(img_in[1][0, 0, :, :], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(img_out[0][0, 0, :, :])
    plt.show()


# print("--------------------------------------------------------------------------")

# for img_in, img_out in dataloader:
#     a = 2