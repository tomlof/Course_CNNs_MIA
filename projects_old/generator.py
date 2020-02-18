#    Copyright 2020 Department of Department of Radiation Sciences, Ume{\aa} University, Ume{\aa}, Sweden
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
#    data generator was based on "https://github.com/ellisdg/3DUnetCNN"


import threading
import os
import glob
import copy
import itertools
import pickle
import ntpath
from random import shuffle
import numpy as np
import nibabel as nib


# =============================================================================================================
# threadsafe
# =============================================================================================================
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


# =============================================================================================================
# 3D
# =============================================================================================================
def get_training_and_validation_and_testing_generators3d(config,
                                                         is_training,
                                                         train_split,
                                                         val_split,
                                                         test_split,
                                                         batch_size,
                                                         n_labels,
                                                         labels=None,
                                                         patch_shape=None,
                                                         skip_blank=True,
                                                         patch_overlap=0,
                                                         cropping_slices=None):
    """
    Creates the training and validation generators that can be used when training the model.
    :param skip_blank: If True, any blank (all-zero) label images/patches will be skipped by the data generator.
    :param patch_shape: Shape of the data to return with the generator. If None, the whole image will be returned.
    (default is None)
    :param labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
    should be equal to the n_labels value.
    :param batch_size: Size of the batches that the training generator will provide.
    :param n_labels: Number of binary labels.
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """

    ids = glob.glob(os.path.join(config["raw_dataset_dir"], "*", "*"))

    train_list, val_list, test_list = get_train_valid_test_split(ids,
                                                                 train_split,
                                                                 val_split,
                                                                 test_split,
                                                                 is_shuffle_data=True)

    make_dir(config["model_path"])

    print("train_list:", train_list)

    print(">> prepare train generator")
    train_generator = data_generator3d(config, train_list)
    print(">> prepare validation generator")
    val_generator = data_generator3d(config, val_list)
    print(">> prepare test generator")
    test_generator = data_generator3d(config, test_list)

    print(">> generate patches and save to disk")

    if is_training:
        print(">> save train...")
        save_all_patches_data_generator(config, train_list)
        print(">> save val...")
        save_all_patches_data_generator(config, val_list)
    else:
        print(">> save test...")
        save_all_patches_data_generator(config, test_list)

    print(">> compute number of training and validation steps")
    num_training_steps = get_number_of_steps(get_number_of_patches3d(config,
                                                                     train_list,
                                                                     patch_shape=config["patch_shape"],
                                                                     patch_overlap=config["patch_overlap"],
                                                                     cropping_slices=config["cropping_slices"]
                                                                     ),
                                             config["batch_size"])
    num_validation_steps = get_number_of_steps(get_number_of_patches3d(config,
                                                                       val_list,
                                                                       patch_shape=config["patch_shape"],
                                                                       patch_overlap=config["patch_overlap"],
                                                                       cropping_slices=config["cropping_slices"]
                                                                       ),
                                               config["batch_size"])

    if is_training:
        print("Number of training steps: ", num_training_steps)
        print("Number of validation steps: ", num_validation_steps)
        return train_generator, val_generator, num_training_steps, num_validation_steps
    else:
        return test_generator


def get_train_valid_test_split(ids, train_split, val_split, test_split, is_shuffle_data=False):
    def get_ids_from_split(ids, split_str):
        split = [int(s) for s in split_str.split(":") if s.isdigit()]
        return ids[split[0]:split[1]]

    if is_shuffle_data:
        shuffle(ids)

    train_list = get_ids_from_split(ids, train_split)
    val_list = get_ids_from_split(ids, val_split)
    test_list = get_ids_from_split(ids, test_split)
    return train_list, val_list, test_list


def get_data_from_file(config, subject_data, index, patch_shape=None):
    if patch_shape:
        patient_path, patch_index = index
        data, truth = get_data_from_file(
            config, subject_data, patient_path, patch_shape=None)
        x = get_patch_from_3d_data(data, patch_shape, patch_index)
        y = get_patch_from_3d_data(truth, patch_shape, patch_index)
    else:
        if len(subject_data) > 2:
            x = np.asarray(subject_data[:-1])
        else:
            x = np.asarray(subject_data[-2])
        y = np.asarray(subject_data[-1], dtype=np.uint8)
    if len(x.shape) == 3:
        x = np.expand_dims(x, axis=0)
    return x, y


def save_all_patches_data_generator(config, index_list):
    if config["patch_shape"] is not None:
        index_list = create_patch_index_list(index_list,
                                             image_shape=config["image_shape"],
                                             patch_shape=config["patch_shape_3d"],
                                             patch_overlap=config["patch_overlap"],
                                             cropping_slices=config["cropping_slices"])

    folder = config["splitted_dataset_dir"]
    make_dir(folder)

    count_is_write = 0
    len_index_list = len(index_list)

    patient_path_prev = subject_data = None
    while len(index_list) > 0:
        index = index_list.pop()
        path_data, path_truth = get_path_patch(
            folder, index, config["patch_shape"])

        patient_path, _ = index
        if patient_path != patient_path_prev:
            images = read_image_files(config, patient_path)
            subject_data = [image.get_fdata() for image in images]
        patient_path_prev = patient_path

        try:
            data, truth = read_pickle(
                path_data), read_pickle(path_truth)
        except:
            data, truth = get_data_from_file(config,
                                             subject_data,
                                             index,
                                             patch_shape=config["patch_shape"])

            count_is_write += 1
            write_pickle(data, path_data)
            write_pickle(truth, path_truth)
            if len(index_list) > 0 and len(index_list) % 100 == 0:
                print(len(index_list))
    print("write: {}/{}".format(count_is_write, len_index_list))


@threadsafe_generator
def data_generator3d(config, index_list):
    orig_index_list = index_list
    while True:
        x_list = list()
        y_list = list()
        index_list = create_patch_index_list(orig_index_list,
                                             image_shape=config["image_shape"],
                                             patch_shape=config["patch_shape"],
                                             patch_overlap=config["patch_overlap"],
                                             cropping_slices=config["cropping_slices"])

        shuffle(index_list)
        while len(index_list) > 0:
            index = index_list.pop()
            add_data3d(x_list,
                       y_list,
                       config,
                       index,
                       patch_shape=config["patch_shape"]
                       )

            if len(x_list) == config["batch_size"] or (len(index_list) == 0 and len(x_list) > 0):
                yield convert_data3d(x_list, y_list, n_labels=config["n_labels"], labels=config["labels"])
                x_list = list()
                y_list = list()


def convert_data3d(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y == 1] = 0
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels3d(y, n_labels=n_labels, labels=labels)
    return x, y


def get_multi_class_labels3d(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y


def augment_data3d(data, truth):
    """ data augumentation """
    # Place your data augmentation implementation here
    # Hint: try tensorlayer or you can implement yourself
    """ data augumentation """
    return data, truth


def normalize_data3d(data):
    """ data augumentation """
    # Place your data augmentation implementation here
    """ data augumentation """
    return data


def add_data3d(x_list, y_list, config, index, patch_shape=None, augment=True):
    """
    Adds data from the data file to the given lists of feature and target data
    :return:
    """
    folder = config["splitted_dataset_dir"]

    path_data, path_truth = get_path_patch(folder, index, patch_shape)

    # we dont need to read all files. If data-truth weren't written to disk => ignore
    try:
        # force to read from pickle
        data, truth = read_pickle(
            path_data), read_pickle(path_truth)

        if augment:
            data_list = list()
            for i in range(data.shape[0]):
                data_list.append(data[i, :, :, :])
            data_list, truth = augment_data3d(data=data_list, truth=truth)
            for i in range(data.shape[0]):
                data[i, :, :, :] = data_list[i]
        truth = truth[np.newaxis]

        x_list.append(data)
        y_list.append(truth)

    except:
        pass


# =============================================================================================================
# utils
# =============================================================================================================
def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def write_pickle(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


def get_path_patch(folder, index, patch_shape):
    idx = get_filename_without_extension(index[0])
    fr = np.ndarray.tolist(index[1])
    patch_shape = list(patch_shape)
    name = "{}-{}-{}".format(str(idx), '_'.join(str(e)
                                                for e in fr), '_'.join(str(e) for e in patch_shape))
    name_data = name + "-data.pickle"
    name_truth = name + "-truth.pickle"
    path_data = os.path.join(folder, name_data)
    path_truth = os.path.join(folder, name_truth)
    return path_data, path_truth


def get_parent_dir(path):
    return os.path.abspath(os.path.join(path, os.pardir))


def get_filename(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_filename_without_extension(path):
    filename = get_filename(path)
    return os.path.splitext(filename)[0]


def read_image_files(config, patient_path):
    image_files = list()
    patient_id = get_filename_without_extension(patient_path)
    modalities = config["training_modalities"] + config["truth"]
    for modality in modalities:
        image_files.append("{}/{}_{}.nii.gz".format(patient_path,
                                                    patient_id,
                                                    modality))
    image_list = list()
    for index, image_file in enumerate(image_files):
        image = read_image(image_file)
        if index < len(modalities) - 1:
            image = normalize_data3d(image)
        image_list.append(image)
    return image_list


def read_image(in_file, crop=None):
    def fix_shape(image):
        if image.shape[-1] == 1:
            return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
        return image
    # print("Reading: {0}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    image = fix_shape(image)
    return image


def compute_patch_indices(image_shape, patch_shape, overlap, cropping_slices, start=None):
    if isinstance(overlap, int):
        overlap = np.asarray([overlap] * len(image_shape))

    image_shape = np.asarray(image_shape)
    patch_shape = np.asarray(patch_shape)
    overlap = np.asarray(overlap)
    image_cropping_slicesshape = np.asarray(cropping_slices)
    if start is None:
        n_patches = np.ceil(image_shape / (patch_shape - overlap))
        overflow = (patch_shape - overlap) * n_patches - image_shape + overlap
        start = -np.ceil(overflow/2)
    elif isinstance(start, int):
        start = np.asarray([start] * len(image_shape))
    stop = image_shape + start
    step = patch_shape - overlap
    return get_set_of_patch_indices(start, stop, step)


def get_set_of_patch_indices(start, stop, step):
    return np.asarray(np.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                               start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=np.int)


def get_patch_from_3d_data(data, patch_shape, patch_index):
    """
    Returns a patch from a numpy array.
    :param data: numpy array from which to get the patch.
    :param patch_shape: shape/size of the patch.
    :param patch_index: corner index of the patch.
    :return: numpy array take from the data with the patch shape specified.
    """
    patch_index = np.asarray(patch_index, dtype=np.int16)
    patch_shape = np.asarray(patch_shape)
    image_shape = data.shape[-3:]
    if np.any(patch_index < 0) or np.any((patch_index + patch_shape) > image_shape):
        data, patch_index = fix_out_of_bound_patch_attempt(
            data, patch_shape, patch_index)
    return data[..., patch_index[0]:patch_index[0]+patch_shape[0], patch_index[1]:patch_index[1]+patch_shape[1],
                patch_index[2]:patch_index[2]+patch_shape[2]]


def fix_out_of_bound_patch_attempt(data, patch_shape, patch_index, ndim=3):
    """
    Pads the data and alters the patch index so that a patch will be correct.
    :param data:
    :param patch_shape:
    :param patch_index:
    :return: padded data, fixed patch index
    """
    image_shape = data.shape[-ndim:]
    pad_before = np.abs((patch_index < 0) * patch_index)
    pad_after = np.abs(((patch_index + patch_shape) > image_shape)
                       * ((patch_index + patch_shape) - image_shape))
    pad_args = np.stack([pad_before, pad_after], axis=1)
    if pad_args.shape[0] < len(data.shape):
        pad_args = [[0, 0]] * \
            (len(data.shape) - pad_args.shape[0]) + pad_args.tolist()
    data = np.pad(data, pad_args, mode="edge")
    patch_index += pad_before
    return data, patch_index


def create_patch_index_list(index_list,
                            image_shape,
                            patch_shape,
                            patch_overlap,
                            cropping_slices=None):
    patch_index = list()
    for index in index_list:
        patches = compute_patch_indices(image_shape, patch_shape,
                                        overlap=patch_overlap,
                                        cropping_slices=cropping_slices)
        patch_index.extend(itertools.product([index], patches))
    return patch_index


def get_number_of_patches3d(config, index_list, patch_shape=None, patch_overlap=0,
                            cropping_slices=None,
                            ):
    if patch_shape:
        index_list = create_patch_index_list(index_list,
                                             image_shape=config["image_shape"],
                                             patch_shape=patch_shape,
                                             patch_overlap=patch_overlap,
                                             cropping_slices=cropping_slices)

        count = 0
        for i, index in enumerate(index_list, 0):
            if i % 50 == 0 and i > 0:
                print(">> processing {}/{}, added {}/{}".format(i,
                                                                len(index_list), count, len(index_list)))
            x_list = list()
            y_list = list()
            add_data3d(x_list, y_list, config, index,
                       patch_shape=patch_shape)
            if len(x_list) > 0:
                count += 1

                # for debuging
                x, y = convert_data3d(
                    x_list, y_list, n_labels=3, labels=(0, 1, 2, 4))

        return count
    else:
        return len(index_list)


def make_dir(dir):
    if not os.path.exists(dir):
        print("-"*60)
        print(">> making dir", dir)
        os.makedirs(dir)


def setup_generator(config, is_training=True, model_dim=3,
                    train_split="0:214", val_split="214:268", test_split="268:335"):
    """
    SETUP BATCHES OF TENSOR IMAGE DATA WITH REAL-TIME DATA AUGMENTATION.
    :param config: Experiment config file.
    :param model_dim: Dimension of segmentation application. For example, U-Net 2D, 3D or 2.5D
        If unspecified, model_dim is set to 3.
    :return: train_generator, validation_generator, n_train_steps, n_validation_steps
    """
    if is_training:
        train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_and_testing_generators3d(
            config,
            is_training,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            batch_size=config["batch_size"],
            n_labels=config["n_labels"],
            labels=config["labels"],
            patch_shape=config["patch_shape"],
            skip_blank=config["skip_blank"],
            patch_overlap=config["patch_overlap"],
            cropping_slices=config["cropping_slices"])
        return train_generator, validation_generator, n_train_steps, n_validation_steps

    else:
        test_generator = get_training_and_validation_and_testing_generators3d(
            config,
            is_training,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            batch_size=config["batch_size"],
            n_labels=config["n_labels"],
            labels=config["labels"],
            patch_shape=config["patch_shape"],
            skip_blank=config["skip_blank"],
            patch_overlap=config["patch_overlap"],
            cropping_slices=config["cropping_slices"])
        return test_generator