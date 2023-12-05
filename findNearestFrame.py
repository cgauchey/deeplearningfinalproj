"""
    Goal of the file:
        - Take a camera predicted class
        - Within that class, find the nearest frame (euclidean distance) to the predicted pose.
        - Return the image of that frame
"""

import argparse
from utils import constants, dataLoad
from models import modelUtils
from models import poseNet
import torch
import numpy as np
import matplotlib.pyplot as plt

def find_nearest(data, pre_class, pred_pose, imshow=False, verbose=False, actual_pose=None):
    # Assume data is a dataset with all images

    # Find all images with the same class as the predicted class
    indices = []
    for i in range(len(data)):
        if int(data[i][1][0]) == pre_class:
            indices.append(i)
    
    if verbose:
        print("\nFound {} images with class {}\n".format(len(indices), pre_class))

    # Get the poses of all the images with the same class as the predicted class
    poses = []
    for i in indices:
        poses.append(data[i][1][1:])
    
    # Find the nearest pose to the predicted pose (and its index)
    nearest_pose = None
    nearest_pose_idx = None
    min_dist = float("inf")
    for i, pose in enumerate(poses):
        dist = torch.dist(pose, pred_pose)
        if dist < min_dist:
            min_dist = dist
            nearest_pose = pose
            nearest_pose_idx = indices[i]
        
    # Now, get the image of the nearest pose (permute to put channels last)
    nearest_img = data[nearest_pose_idx][0]
    nearest_img = nearest_img.permute(1, 2, 0)
    if imshow:
        # Display the image non-blocking
        plt.imshow(nearest_img)
        plt.show(block=False)
        plt.title("Nearest Frame to Predicted Pose")
        plt.pause(0.001)


    if verbose:
        print("\nPredicted class: {}\n".format(pre_class))

        # Break pose in to two parts (x, y, z) and (rx, ry, rz)
        pred_pose_np = pred_pose.detach().numpy()
        nearest_pose_np = nearest_pose.detach().numpy()

        pred_pose_pos = pred_pose_np[:3]
        pred_pose_rot = pred_pose_np[3:]

        nearest_pose_pos = nearest_pose_np[:3]
        nearest_pose_rot = nearest_pose_np[3:]

        if actual_pose is not None:
            actual_pose_np = actual_pose.detach().numpy()
            actual_pose_pos = actual_pose_np[:3]
            actual_pose_rot = actual_pose_np[3:]

            print("True position: {}".format(actual_pose_pos))
            print("Predicted position: {}".format(pred_pose_pos))
            print("Nearest position: {}\n".format(nearest_pose_pos))

            print("True rotation: {}".format(actual_pose_rot))
            print("Predicted rotation: {}".format(pred_pose_rot))
            print("Nearest rotation: {}\n".format(nearest_pose_rot))

            # Print out the distance to the nearest versus to the actual
            actual_to_pred_dist = torch.dist(actual_pose, pred_pose)
            actual_to_nearest_dist = torch.dist(actual_pose, nearest_pose)
            pred_to_nearest_dist = torch.dist(pred_pose, nearest_pose)

            print("Distance between predicted pose and true pose: {}".format(actual_to_pred_dist))
            print("Distance between nearest pose and true pose: {}".format(actual_to_nearest_dist))
            print("Distance between predicted pose and nearest pose: {}\n".format(pred_to_nearest_dist))
        else:
            print("Predicted position: {}".format(pred_pose_pos))
            print("Nearest position: {}\n".format(nearest_pose_pos))

            print("Predicted rotation: {}".format(pred_pose_rot))
            print("Nearest rotation: {}\n".format(nearest_pose_rot))

    return nearest_img

if __name__ == "__main__":
    # Fill in these values and run
    # pred_class = 1
    # pred_pose = torch.tensor([-0.38865992, -0.5422972, -0.08911549, 1.4426252, -0.345953173, -1.3733114])
    # actual_pose = torch.tensor([-0.55068116, -0.4394174, 0.01071588, 1.73502943, -1.26409687, -2.71521486])

    pred_class = 2
    pred_pose = torch.tensor([0.13467625, 0.45333573, 0.25124094, -0.7113406, 0.7395303, -1.5051279])
    actual_pose = torch.tensor([-0.03780979, 0.55761269, 0.04036432, -1.44847438, -0.37112511, -1.63266237])

    # Load the data
    data = dataLoad.load_data(constants.DATA_IMGS_DIR_PROCESSED, constants.DATA_SFM_DIR, verbose=True)

    # Find nearest
    img = find_nearest(data, pred_class, pred_pose, imshow=True, verbose=True, actual_pose=actual_pose)

    # Pause for image
    input("Press Enter to exit ...")
