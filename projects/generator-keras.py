# Import necessary packages for loading the dataset

import numpy as np  # Package for matrix operations, handling data
np.random.seed(2020)
import os
# import cv2
import matplotlib.pyplot as plt  # Package for plotting
#from PIL import Image
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.utils import to_categorical
import tensorflow.keras as keras


class DataGenerator(keras.utils.Sequence):

    def __init__(self,
                 data_path,
                 inputs,
                 outputs,
                 batch_size=32):

        self.data_path = data_path
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size

        if data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(data_path):
            raise ValueError('The data path is incorrectly defined.')

        self.file_idx = 0
        self.file_list = [self.data_path + '/' + s
                          for s in os.listdir(self.data_path)]

        self.on_epoch_end()
        with np.load(self.file_list[0]) as npzfile:
            self.out_dims = []
            self.in_dims = []
            self.n_channels = 1

            for i in range(len(self.inputs)):
                im = npzfile[self.inputs[i]]
                self.in_dims.append((self.batch_size,
                                     *np.shape(im),
                                     self.n_channels))

            for i in range(len(self.outputs)):
                im = npzfile[self.outputs[i]]
                self.out_dims.append((self.batch_size,
                                      *np.shape(im),
                                      self.n_channels))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.file_list)) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) *
                               self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.file_list[k] for k in indexes]

        # Generate data
        i, o = self.__data_generation(list_IDs_temp)

        return i, o

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))
        np.random.shuffle(self.indexes)

    #@threadsafe_generator
    def __data_generation(self, temp_list):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        inputs = []
        outputs = []

        for i in range(len(self.inputs)):
            inputs.append(np.empty(self.in_dims[i]).astype(np.float32))

        for i in range(self.outputs.__len__()):
            outputs.append(np.empty(self.out_dims[i]).astype(np.float32))

        for i, ID in enumerate(temp_list):
            with np.load(ID) as npzfile:

                for idx in range(len(self.inputs)):
                    x = npzfile[self.inputs[idx]].astype(np.float32)
                    x = x[..., np.newaxis]
                    inputs[idx][i, ...] = x

                for idx in range(len(self.outputs)):
                    x = npzfile[self.outputs[idx]].astype(np.float32)
                    x = x[..., np.newaxis]
                    outputs[idx][i, ...] = x

        return inputs, outputs


gen_dir = "/Home/guests/guest1/Documents/data/"
# gen_dir = "/import/software/3ra023vt20/brats/data/"

# Available arrays in data: 'flair', 't1', 't2', 't1ce', 'mask'
# See the lab instructions for more info about the arrays
input_arrays = ['flair', 't1', 't1ce']
output_arrays = ['mask']
batch_size = 48
gen_train = DataGenerator(data_path=gen_dir + 'training',
                          inputs=input_arrays,
                          outputs=output_arrays,
                          batch_size=batch_size)

# Look at some sample images
img_in, img_out = gen_train[np.random.randint(0, len(gen_train))]
for inp in range(np.shape(img_in)[0]):
    plt.figure(figsize=(12, 5))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img_in[inp][i, :, :, 0])
        plt.title('Image size: ' + str(np.shape(img_in[inp][i, :, :, 0])))
        plt.tight_layout()
    plt.suptitle('Input for array: ' + gen_train.inputs[inp])
    plt.show()

plt.figure(figsize=(12, 4))
for outp in range(np.shape(img_out)[0]):
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img_out[outp][i, :, :, 0])
        plt.title('Image size: ' + str(np.shape(img_out[outp][i, :, :, 0])))
        plt.tight_layout()

    plt.suptitle('Output for array: ' + gen_train.outputs[outp])
    plt.show()
