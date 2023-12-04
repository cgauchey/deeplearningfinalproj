import os
import torch

# All paths are from the root directory of the project
DATA_SFM_DIR = os.path.join('data', 'SfM')
DATA_IMGS_DIR = os.path.join(DATA_SFM_DIR, 'images')
DATA_SPARSE_DIR = os.path.join(DATA_SFM_DIR, 'sparse')
DATA_CAMS_FILE_DIR = os.path.join(DATA_SPARSE_DIR, 'cameras.bin')
DATA_IMGS_FILE_DIR = os.path.join(DATA_SPARSE_DIR, 'images.bin')
DATA_POINTS_FILE_DIR = os.path.join(DATA_SPARSE_DIR, 'points3D.bin')

# NOTE: not currently stored in project repo
# DATA_IMGS_DIR_PROCESSED = os.path.join('data', 'processedImages')
DATA_IMGS_DIR_PROCESSED = os.path.join('data', '0000_0524')

DEFAULT_MODEL_SAVE_FOLDER = os.path.join('data', 'savedModels')

# Determine device for model


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
