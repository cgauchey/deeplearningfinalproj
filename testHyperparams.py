import subprocess
import argparse
from utils import constants, dataLoad
from models import modelUtils, poseNet
from matplotlib import pyplot as plt
import datetime
import torch
from torch import optim
import os

def testHyperparams():

    # Normal training parameters
    model_save_folder = constants.DEFAULT_MODEL_SAVE_FOLDER
    save_freq = 10
    train_verbose = True
    print_freq = 5
    image_folder = constants.DATA_IMGS_DIR_PROCESSED
    model_file = constants.DATA_SFM_DIR
    seed = 42
    log_path = constants.DEFAULT_LOGS_FOLDER

    # Load the data
    print("Loading data...")
    data = dataLoad.load_data(image_folder, model_file, verbose=train_verbose)

    print("Making train/val/test split...")
    train_dataset, val_dataset, test_dataset = dataLoad.make_train_val_test_split(data, seed, verbose=train_verbose)

    device = constants.get_device()

    # Hyperparameter values we will test
    # epochs = [20, 50]
    # patiences = [5, 10]
    # batch_sizes = [32, 64, 256]
    # learning_rates = [0.001, .005]

    epochs = [5]
    patiences = [5]
    batch_sizes = [32, 64]
    learning_rates = [0.001]

    # Keep track of the losses with the hyperparameters
    train_losses = {}
    val_losses = {}

    # Testing loop
    for e in epochs:
        for p in patiences:
            for bs in batch_sizes:
                for lr in learning_rates:
                    # Make the model
                    NN = poseNet.ClassyPoseNet(constants.MODEL_FEATURE_DIM, constants.MODEL_DROPOUT_RATE, constants.MODEL_NUM_CLASSES, device).to(device)
                    
                    # Make the optimizer
                    optimizer = optim.Adam(NN.parameters(), lr=lr)

                    # Train the model
                    print("Training model with epochs={}, patience={}, batch_size={}, learning_rate={}".format(e, p, bs, lr))
                    best_model, best_epoch_num, train_losses[(e,p,bs,lr)], val_losses[(e, p, bs, lr)] = modelUtils.train(model=NN, 
                                                                                                            optimizer=optimizer,
                                                                                                            train_dataset=train_dataset, 
                                                                                                            val_dataset=val_dataset, 
                                                                                                            epochs=e, 
                                                                                                            batch_size=bs, 
                                                                                                            patience=p, 
                                                                                                            save_freq=save_freq, 
                                                                                                            seed=seed,
                                                                                                            print_freq=print_freq, 
                                                                                                            verbose=train_verbose,
                                                                                                            model_save_folder=model_save_folder,
                                                                                                            logfolder=log_path)    

    # Plot all the losses
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].set_title("Training Losses")
    ax[1].set_title("Validation Losses")
    for e in epochs:
        for p in patiences:
            for bs in batch_sizes:
                for lr in learning_rates:
                    ax[0].plot(train_losses[(e,p,bs,lr)], label="epochs={}, patience={}, batch_size={}, learning_rate={}".format(e, p, bs, lr))
                    ax[1].plot(val_losses[(e,p,bs,lr)], label="epochs={}, patience={}, batch_size={}, learning_rate={}".format(e, p, bs, lr))

    # The legends are going to be very long, since there are many different hyperparameter combinations
    # So we'll put the legend outside the plot so that it doesn't cover the plot
    ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=5)
    ax[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=5)
    

    # Create timestamp for saving the plot
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    plt.savefig(os.path.join(log_path, "hyperparam_loss_plot_{}.png".format(timestamp)))

    # Print the path to saved fiture
    curr_path = os.path.abspath(os.getcwd())
    print("Plot saved to {}".format(os.path.join(curr_path, log_path, "hyperparam_loss_plot_{}.png".format(timestamp))))


if __name__ == '__main__':
    testHyperparams()