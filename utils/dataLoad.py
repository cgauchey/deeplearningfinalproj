import numpy as np
import torch
from utils import constants, colmapUtils
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import struct
import collections
import sys
import cv2
import os
from PIL import Image
import math
from sklearn.model_selection import train_test_split

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


def load_data(img_folder_path=constants.DATA_IMGS_DIR_PROCESSED, 
              models_folder_path=constants.DATA_SFM_DIR,
              transformation=transform_to_float_and_channels, verbose=False, min_cams=100):
    
    # Store all data here
    X_list = []
    y_list = []

    # Data is stored in COLMAP binary files
    model_idx = -1
    print(os.getcwd())
    for split in sorted(os.listdir(models_folder_path)):
        for model_num in os.listdir(os.path.join(models_folder_path, split, 'sparse')):

            if model_num.endswith('.bin'):
                continue

            # Extract data
            img_bin = os.path.abspath(os.path.join(split, 'sparse', model_num, 'images.bin'))

            # Load image information
            # Each BaseImage has qvec (rotation), name (img filename), tvec (translation), 
            #   camera_id (camera model), xys (x and y values)
            imageBase = colmapUtils.read_extrinsics_binary(img_bin)

            # Skip models with too few cameras
            if len(imageBase.keys()) < min_cams:
                continue
            model_idx += 1

            # Logging
            if verbose:
                print(f'Read cameras from {img_bin}')

            # Save images, rotations and translations to lists
            images = []
            q_rots = []
            trans = []

            for i, key in enumerate(imageBase.keys()):

                # Extract extrinsic info for the given camera
                iBase = imageBase[key]

                # Logging
                if verbose and i % 10 == 0:
                    print("Loading image: {}/{}".format(i+1, len(imageBase.keys())))

                # Get the image from the img_folder
                img_path = os.path.join(img_folder_path, iBase.name)
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

            # The input to this regression problem is the image
            X = images
            X_list.append(X)

            # The output has three components: the model index, the camera euler rotations, and the camera spatial coordinates
            idx = torch.full((len(imageBase.keys()), 1), model_idx)
            y = torch.cat([idx, euler_rots, trans], dim=1)
            y_list.append(y)

    # Now, make this into a dataset
    all_X = torch.cat(X_list, dim=1)
    all_y = torch.cat(y_list, dim=1)
    dataset = TensorDataset(all_X, all_y)

    # Use the custom dataset wrapper to apply transformation
    dataset_with_transform = CustomDataset(dataset, range(len(dataset)), transform=transformation)

    return dataset_with_transform

def make_train_val_test_split(dataset, seed, verbose=False):
    # Make 80-10-10 train/val/test split of dataset

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Get the indices for the subsets
    indices = list(range(len(dataset)))
    train_indices, val_test_indices = train_test_split(indices, 
                                                        test_size=0.2, 
                                                        random_state=seed)
    
    val_indices, test_indices = train_test_split(val_test_indices,
                                                    test_size=0.5,
                                                    random_state=seed)
    
    if verbose:
        print("Train size: {}".format(len(train_indices)))
        print("Val size: {}".format(len(val_indices)))
        print("Test size: {}".format(len(test_indices)))
    
    # Make the subsets
    train_dataset = CustomDataset(dataset, train_indices)
    val_dataset = CustomDataset(dataset, val_indices)
    test_dataset = CustomDataset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset
                                    