import argparse
from utils import constants, dataLoad
from models import modelUtils, poseNet
from matplotlib import pyplot as plt
import datetime
import torch
from torch import optim
import os

def run_training():
    # Takes command line arguments, then trains model
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=constants.DEFAULT_MODEL_SAVE_FOLDER, help="Path to save model weights to")
    parser.add_argument("-e", "--epochs", default=20, help="Number of epochs to train for")
    parser.add_argument("-b", "--batch_size", default=32, help="Batch size to use for training")
    parser.add_argument("-p", "--patience", default=5, help="Number of epochs to wait before early stopping")
    parser.add_argument("-lr", "--learning_rate", default=5, help="Learning rate for the optimizer")
    parser.add_argument("-s", "--save_freq", default=5, help="Number of epochs between each model save")
    parser.add_argument("-v", "--verbose", action='store_true', help="Whether to print out training progress")
    parser.add_argument("-i", "--image_folder", default=constants.DATA_IMGS_DIR_PROCESSED, help="Path to folder containing processed images")
    parser.add_argument("-f", "--model_file", default=constants.DATA_SFM_DIR, help="Path to file containing SfM files")
    parser.add_argument("-se", "--seed", default=42, help="Random seed to use for training")
    parser.add_argument("-pf", "--print_freq", default=1, help="Number of epochs between each print statement")
    parser.add_argument("-l", "--log", default=False, action='store_true', help="Path to folder to save log files to")
    args = parser.parse_args()

    # Convert to correct types
    args.model = str(args.model)
    args.epochs = int(args.epochs)
    args.batch_size = int(args.batch_size)
    args.patience = int(args.patience)
    args.learning_rate = float(args.learning_rate)
    args.save_freq = int(args.save_freq)
    args.verbose = bool(args.verbose)
    args.image_folder = str(args.image_folder)
    args.model_file = str(args.model_file)
    args.seed = int(args.seed)
    args.print_freq = int(args.print_freq)
    args.log = bool(args.log)

    if args.log:
        log_path = constants.DEFAULT_LOGS_FOLDER
    else:
        log_path = None
    
    # Load the data
    if args.verbose:
        print("Loading data...")
    data = dataLoad.load_data(args.image_folder, args.model_file, verbose=args.verbose)

    # Make 80-10-10 train/val/test split of dataset
    if args.verbose:
        print("Making train/val/test split...")
    train_dataset, val_dataset, test_dataset = dataLoad.make_train_val_test_split(data, args.seed, verbose=args.verbose)

    # Ensure we're using the GPU
    device = constants.get_device()

    # Create the PyTorch model
    NN = poseNet.ClassyPoseNet(constants.MODEL_FEATURE_DIM, constants.MODEL_DROPOUT_RATE, constants.MODEL_NUM_CLASSES, device).to(device)

    # Create the optimizer
    optimizer = optim.Adam(NN.parameters(), lr=args.learning_rate)

    # Train the model
    if args.verbose:
        print("Training model...")
    best_model, best_epoch_num, train_losses, val_losses = modelUtils.train(model=NN, 
                                                                            optimizer=optimizer,
                                                                            train_dataset=train_dataset, 
                                                                            val_dataset=val_dataset, 
                                                                            epochs=args.epochs, 
                                                                            batch_size=args.batch_size, 
                                                                            patience=args.patience, 
                                                                            save_freq=args.save_freq, 
                                                                            seed=args.seed,
                                                                            print_freq=args.print_freq, 
                                                                            verbose=args.verbose,
                                                                            model_save_folder=args.model,
                                                                            logfolder=log_path)

    # Test the model
    if args.verbose:
        print("Testing model...")
    avg_test_loss = modelUtils.evaluate_model(best_model, test_dataset, batch_size=args.batch_size, verbose=args.verbose, logfolder=log_path)

    print("Training loss of best model: {}".format(train_losses[best_epoch_num]))
    print("Validation loss: {}".format(val_losses[best_epoch_num]))
    print("Test loss: {}".format(avg_test_loss))

    # Plot the losses and save them into the model save folder
    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Create a timestamp for the file name
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    plt.savefig(os.path.join(args.model, f"_losses_{timestamp}.png"))

    if args.verbose:
        # Print out the full save path 
        curr_path = os.path.abspath(os.getcwd())
        print("Plot saved to {}".format(os.path.join(curr_path, args.model, f"_losses_{timestamp}.png")))


if __name__ == "__main__":
    run_training()