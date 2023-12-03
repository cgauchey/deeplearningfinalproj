# Similar to trainModel.py, this script is used to evaluate the performance of the trained model. 
# It will load in model weights from the location specified by the --model argument, and then evaluate the model on the test set.

import argparse
from utils import constants, dataLoad
from models import modelUtils

def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate a model on the test set")
    parser.add_argument("-m", "--model",  required=True, help="Path to model weights to load")
    parser.add_argument("-b", "--batch_size", default=32, help="Batch size to use for training")
    parser.add_argument("-v", "--verbose", default=True, help="Whether to print out training progress")
    parser.add_argument("-i", "--image_folder", default=constants.DATA_IMGS_DIR_PROCESSED, help="Path to folder containing processed images")
    parser.add_argument("-l", "--label_file", default=constants.DATA_IMGS_FILE_DIR, help="Path to file containing image labels")
    parser.add_argument("-se", "--seed", default=42, help="Random seed to use for training")
    args = parser.parse_args()

    # Convert to correct types
    args.model = str(args.model)
    args.batch_size = int(args.batch_size)
    args.verbose = bool(args.verbose)
    args.image_folder = str(args.image_folder)
    args.label_file = str(args.label_file)
    args.seed = int(args.seed)
    
    # Load the data
    if args.verbose:
        print("Loading data...")
    data = dataLoad.load_data(args.image_folder, args.label_file, verbose=args.verbose)

    # Make 80-10-10 train/val/test split of dataset
    # Even though we are not training, we still want to do this so that we are not evaluating it on the data it trained on
    if args.verbose:
        print("Making train/val/test split...")
    train_dataset, val_dataset, test_dataset = dataLoad.make_train_val_test_split(data, args.seed, verbose=args.verbose)

    # Get the device
    device = constants.get_device()

    # Load the model
    if args.verbose:
        print("Loading model...")
    model = modelUtils.load_model(args.model, feature_dims=constants.FEATURE_DIMS, dropout_rate=constants.DROPOUT_RATE, device=device, verbose=args.verbose)

    # Test the model
    if args.verbose:
        print("Testing model...")
    avg_test_loss = modelUtils.evaluate_model(model, test_dataset, batch_size=args.batch_size, verbose=args.verbose)

    print("Average test loss: {}".format(avg_test_loss))

if __name__ == "__main__":
    run_evaluation()