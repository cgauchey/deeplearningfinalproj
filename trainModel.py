import argparse
from utils import constants, dataLoad
from models import modelUtils
from matplotlib import pyplot as plt
import datetime

def run_training():
    # Takes command line arguments, then trains model
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=constants.DEFAULT_MODEL_SAVE_FOLDER, help="Path to save model weights to")
    parser.add_argument("-e", "--epochs", default=20, help="Number of epochs to train for")
    parser.add_argument("-b", "--batch_size", default=32, help="Batch size to use for training")
    parser.add_argument("-p", "--patience", default=5, help="Number of epochs to wait before early stopping")
    parser.add_argument("-s", "--save_freq", default=5, help="Number of epochs between each model save")
    parser.add_argument("-v", "--verbose", default=False, help="Whether to print out training progress")
    parser.add_argument("-i", "--image_folder", default=constants.DATA_IMGS_DIR_PROCESSED, 
                        help="Path to folder containing processed images")
    # parser.add_argument("-l", "--label_file", default=constants.DATA_IMGS_FILE_DIR, help="Path to file containing image labels")
    parser.add_argument("-se", "--seed", default=42, help="Random seed to use for training")
    args = parser.parse_args()

    # Convert to correct types
    args.model = str(args.model)
    args.epochs = int(args.epochs)
    args.batch_size = int(args.batch_size)
    args.patience = int(args.patience)
    args.save_freq = int(args.save_freq)
    args.verbose = bool(args.verbose)
    args.image_folder = str(args.image_folder)
    args.label_file = str(args.label_file)
    args.seed = int(args.seed)

    # Load the data
    if args.verbose:
        print("Loading data...")
    data = dataLoad.load_data(args.image_folder, args.label_file, verbose=args.verbose)

    # Make 80-10-10 train/val/test split of dataset
    if args.verbose:
        print("Making train/val/test split...")
    train_dataset, val_dataset, test_dataset = dataLoad.make_train_val_test_split(data, args.seed, verbose=args.verbose)

    # Train the model
    if args.verbose:
        print("Training model...")
    best_model, best_epoch_num, train_losses, val_losses = modelUtils.train(model_save_folder=args.model, 
                                                                            train_dataset=train_dataset, 
                                                                            val_dataset=val_dataset, 
                                                                            epochs=args.epochs, 
                                                                            batch_size=args.batch_size, 
                                                                            patience=args.patience, 
                                                                            save_freq=args.save_freq, 
                                                                            seed=args.seed, 
                                                                            verbose=args.verbose)

    # Test the model
    if args.verbose:
        print("Testing model...")
    avg_test_loss = modelUtils.test(best_model, test_dataset, batch_size=args.batch_size, verbose=args.verbose)

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
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(args.model + "/losses_" + timestamp + ".png")

if __name__ == "__main__":
    run_training()