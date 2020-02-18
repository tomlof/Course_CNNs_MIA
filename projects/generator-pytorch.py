import tensorflow.keras as keras
import os
import numpy as np
import matplotlib.pyplot as plt


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 data_path,
                 inputs,
                 outputs,
                 batch_size=32,
                 shuffle=True
                 ):
        self.data_path = data_path
        self.inputs = inputs
        self.outputs = outputs
        self.batch_size = batch_size
        self.shuffle = shuffle
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
                self.in_dims.append((self.batch_size,
                                     *np.shape(im),
                                     self.n_channels))
            for i in range(len(self.outputs)):
                im = npzfile[self.outputs[i]]
                self.out_dims.append((self.batch_size,
                                      *np.shape(im),
                                      self.n_channels))
            npzfile.close()

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
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
    # @threadsafe_generator

    def __data_generation(self, temp_list):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
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
                    x = np.expand_dims(x, axis=2)
                    inputs[idx][i, ] = x
                for idx in range(len(self.outputs)):
                    x = npzfile[self.outputs[idx]] \
                        .astype(np.single)
                    x = np.expand_dims(x, axis=2)
                    outputs[idx][i, ] = x
                npzfile.close()
        return inputs, outputs


gen_dir = '/mnt/4a39cb60-7f1f-4651-81cb-029245d590eb/BRATSs/'
gen = DataGenerator(data_path=gen_dir+'training',
                    inputs=['flair', 't1'],
                    outputs=['mask'],
                    batch_size=16,
                    shuffle=True)

while True:
    img_in, img_out = gen.__getitem__(np.random.randint(0, gen.__len__()))
    plt.subplot(1, 3, 1)
    plt.imshow(img_in[0][0, :, :, 0], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(img_in[1][0, :, :, 0], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(img_out[0][0, :, :, 0])
    plt.show()
