import os
import generator as generator
import model as model


# =============================================================================================================
# main
# =============================================================================================================
def init_config():
    config = dict()

    # ===============================================================================
    # Paths
    # ===============================================================================
    config["base"] = "/home/minhvu/github/Course_CNNs_MIA/projects"
    config["raw_dataset_dir"] = os.path.join(config["base"], "data/raw")
    config["id_dataset_path"] = os.path.join(config["base"], "data/ids.txt")
    config["splitted_dataset_dir"] = os.path.join(
        config["base"], "data/splitted")
    config["model_file"] = os.path.join(
        config["base"], "data/model/best_model.h5")

    # ===============================================================================
    # Project description
    # ===============================================================================
    config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
    config["training_modalities"] = ["t1"]
    config["truth"] = ["seg"]

    # ===============================================================================
    # Training configurations
    # ===============================================================================
    # the label numbers to be trained
    config["labels"] = (1, 2, 4)
    # number of labels to be trained
    config["n_labels"] = len(config["labels"])
    # default batch size for 2d network
    config["batch_size_2d"] = 16
    # default batch size for 3d network
    config["batch_size_3d"] = 1

    # ------------------------------------------------------------------------------
    # do not modify these unless you know what you are doing
    # ------------------------------------------------------------------------------
    # workers: Integer. Maximum number of processes to spin up when using process-based threading.
    # If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
    config["workers"] = 4
    # Integer. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
    config["max_queue_size"] = 4
    # Boolean. If True, use process-based threading. If unspecified, use_multiprocessing
    # will default to False. Note that because this implementation relies on multiprocessing,
    # you should not pass non-picklable arguments to the generator as
    # they can't be passed easily to children processes.
    config["use_multiprocessing"] = True

    # ===============================================================================
    # Image configurations
    # ===============================================================================
    # image shape
    config["image_shape"] = (240, 240, 155)
    # default patch shape for 2d network
    config["patch_shape_2d"] = (240, 240, 1)
    # default patch shape for 3d network
    config["patch_shape_3d"] = (64, 64, 64)

    # ===============================================================================
    # Patch extraction for training configurations
    # ===============================================================================
    # if True, then patches without any target will be skipped
    config["batch_size"] = config["batch_size_3d"]
    config["patch_shape"] = config["patch_shape_3d"]
    config["cropping_slices"] = (0, 0, 0)
    config["patch_overlap"] = (0, 0, 0)
    config["skip_blank"] = True

    # ===============================================================================
    # Training config_commonurations
    # ===============================================================================
    # cutoff the training after this many epochs
    config["n_epochs"] = 200
    # learning rate will be reduced after this many epochs if the validation loss is not improving
    config["patience"] = 5
    # training will be stopped after this many epochs without the validation loss improving
    config["early_stop"] = 12
    # initial learning rate
    config["initial_learning_rate"] = 1e-4
    # factor by which the learning rate will be reduced
    config["learning_rate_drop"] = 0.2

    return config


def main():

    config = init_config()
    unet = model.unet(input_size=(64, 64, 64))
    train_generator, validation_generator, n_train_steps, n_validation_steps = generator.setup_generator(config,
                                                                                                         is_training=True,
                                                                                                         train_split="0:5",
                                                                                                         val_split="5:8",
                                                                                                         test_split="8:10")
    model.do_training(unet,
                      train_generator,
                      validation_generator,
                      n_train_steps,
                      n_validation_steps,
                      config)


if __name__ == "__main__":
    main()
