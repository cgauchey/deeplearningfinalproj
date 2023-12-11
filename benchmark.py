# This loads the dataset, gets the test data, and then evaluates the loss of a random prediction.

import argparse
from utils import constants, dataLoad
from models import modelUtils, poseNet
from matplotlib import pyplot as plt
import datetime
import torch
from torch import optim
import os

def testRandomModel():
    print_freq = 5
    seed = 42
    image_folder = constants.DATA_IMGS_DIR_PROCESSED
    model_file = constants.DATA_SFM_DIR

    # Load the data
    print("Loading data...")
    data = dataLoad.load_data(image_folder, model_file, verbose=True)

    print("Making train/val/test split...")
    train_dataset, val_dataset, test_dataset = dataLoad.make_train_val_test_split(data, seed, verbose=True)

    # Our model will just pick a random class, and predict the origin for the pose (0,0,0,0,0,0)
    # Then run the loss on the test dataset for this

    # Get the test dataset length
    test_dataset_len = len(test_dataset)

    # Make a same length random class prediction (10 classes)
    random_class_pred = torch.randint(0, 10, (test_dataset_len,))

    # Make it one hot
    random_class_pred_one_hot = torch.nn.functional.one_hot(random_class_pred, num_classes=10).float()

    # Make a same length random pose prediction (0,0,0,0,0,0)
    random_pose_pred = torch.zeros(test_dataset_len, 6)

    # Concat
    random_pred = torch.cat((random_class_pred_one_hot, random_pose_pred), dim=1)

    # Get the test dataset labels
    # Having trouble loading the labels from the dataset, lets do it in batches with data loaders
    total_loss = 0
    batch_size = 32
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for i, (X, y), in enumerate(test_dataset_loader):
        # Get the first batch of fake predictions
        random_pred_batch = random_pred[i*batch_size:(i+1)*batch_size]

        # Run the loss
        loss = modelUtils.compute_loss(random_pred_batch, y, 10)

        # Add to total loss
        total_loss += loss.item()

    # Get the average loss
    loss = total_loss / len(test_dataset_loader)
    
    print("Random model loss: {}".format(loss))

if __name__ == "__main__":
    testRandomModel()