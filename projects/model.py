import math
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from functools import partial


def unet(input_size=(64, 64, 64)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy', metrics=[dice_coefficient])

    # model.summary()

    return model


def dice_coefficient(y_true, y_pred, smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None):
    callbacks = list()

    # model checkpoint best
    callbacks.append(ModelCheckpoint(model_file,
                                     save_best_only=True))

    # patience
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity,
                                       patience=early_stopping_patience))
    return callbacks


def do_training(model, train_generator, validation_generator,
                n_train_steps, n_validation_steps, config,
                workers=1, max_queue_size=10, use_multiprocessing=False):
    """
    PERFORM TRAINING.
    :param model: Compiled Keras model. For example, U-Net 2D
    :param train_generator: Training data generator.
    :param validation_generator: Validation data generator.
    :param n_train_steps: Integer. Total number of steps (batches of samples)
        to yield from generator before declaring one epoch finished and starting the next epoch.
        It should typically be equal to ceil(num_samples / batch_size)
        Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
    :param n_validation_steps: Only relevant if validation_data is a generator.
        Total number of steps (batches of samples) to yield from validation_data generator
        before stopping at the end of every epoch. It should typically be equal to the number
        of samples of your validation dataset divided by the batch size.
        Optional for Sequence: if unspecified, will use the len(validation_data) as a number of steps.
    :param config: Experiment config file.
    :param project_name: Comet-ml project name. Send your experiment to a specific project.
        Otherwise will be sent to Uncategorized Experiments.
        If project name does not already exists Comet.ml will create a new project.
    :param workers: Integer. Maximum number of processes to spin up when using process-based threading.
        If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
    :param max_queue_size:  Integer. Maximum size for the generator queue.
        If unspecified, max_queue_size will default to 10.
    :param use_multiprocessing: Boolean. If True, use process-based threading.
        If unspecified, use_multiprocessing will default to False.
        Note that because this implementation relies on multiprocessing, you should not pass non-picklable
        arguments to the generator as they can't be passed easily to children processes.
    """

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=n_train_steps,
                                  epochs=config["n_epochs"],
                                  validation_data=validation_generator,
                                  validation_steps=n_validation_steps,
                                  verbose=2,
                                  workers=config["workers"],
                                  max_queue_size=config["max_queue_size"],
                                  use_multiprocessing=config["use_multiprocessing"],
                                  callbacks=get_callbacks(config["model_file"],
                                                          initial_learning_rate=config["initial_learning_rate"],
                                                          learning_rate_drop=config["learning_rate_drop"],
                                                          learning_rate_patience=config["patience"],
                                                          early_stopping_patience=config["early_stop"]))
