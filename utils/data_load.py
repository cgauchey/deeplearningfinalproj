import numpy as np
import torch
from utils import constants, colmap_utils
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import struct
import collections
import sys
import cv2
import os
from PIL import Image
import math

# Transformation for img to float tensor
class ToTensorForImage(torch.nn.Module):
    def forward(self, x):
        return x.float() / 255

transpose = transforms.Lambda(lambda x: x.transpose(0, 1).transpose(0, 2))

# Make a transoformation to apply to images 
# This one just converts to a float and changes to (channels, height, width) format
transform_to_float_and_channels = transforms.Compose([
    transpose,
    ToTensorForImage(),
])

# Make a custom dataset for the images so we can apply the transformation
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Load and preprocess the image
        sample, label = self.dataset[self.indices[idx]]

        # Apply the custom transformation if available
        if self.transform:
            sample = self.transform(sample)

        return sample, label
    
def batch_quaternion_to_euler(quaternions):
    """
    Convert a batch of quaternions to Euler angles (pitch, roll, yaw).

    Parameters:
    - quaternions (torch.Tensor): Batch of quaternions with shape (batch_size, 4).

    Returns:
    - torch.Tensor: Batch of Euler angles with shape (batch_size, 3) in radians.
    """
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x**2 + y**2)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * math.pi / 2, torch.asin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y**2 + z**2)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([pitch, roll, yaw], dim=1)


def load_data(img_folder_path=constants.DATA_IMGS_DIR, img_file_path=constants.DATA_IMGS_FILE_DIR, transformation=transform_to_float_and_channels, verbose=False):
    # Data is stored using COLMAP method.  The data is stored in bin files.

    img_folder = img_folder_path
    img_bin = img_file_path

    # Load image information
    # Each BaseImage has qvec (rotation), name (img filename), tvec (translation), camera_id (camera model), xys (x and y values)
    imageBase = colmap_utils.read_extrinsics_binary(img_bin)

    # Save images, rotations and translations to lists
    images = []
    q_rots = []
    trans = []

    for i, key in enumerate(imageBase.keys()):
        if verbose and i % 10 == 0:
            print("Loading image: {}/{}".format(i+1, len(imageBase.keys())))

        iBase = imageBase[key]

        # Get the image from the img_folder
        img_path = os.path.join(img_folder, iBase.name)
        img = torch.tensor(np.array(Image.open(img_path)), dtype=torch.uint8)

        # Get the 6DOF vector (rotation and translation)
        # First extract the rotation in Quaternion form
        qvec = iBase.qvec

        # Now extract the position
        tvec = iBase.tvec

        # Add to lists
        images.append(img)
        q_rots.append(qvec)
        trans.append(tvec)

    # Now, convert the lists to tensors
    images = torch.stack(images)
    q_rots = torch.stack(q_rots)
    trans = torch.stack(trans)

    # convert q_rots to euler rotations
    euler_rots = batch_quaternion_to_euler(q_rots)

    # This is going to be a regression problem, where the input is the image, and the output is the 6DOF vector
    X = images

    # For y, combine the euler rotations and the translations to get something in shape (dataset_sie, 6)
    y = torch.cat([euler_rots, trans], dim=1)

    # Now, make this into a dataset
    dataset = TensorDataset(X, y)

    # Use the custom dataset wrapper to apply transformation
    dataset_with_transform = CustomDataset(dataset, range(len(dataset)), transform=transformation)

    return dataset_with_transform

