import os

# All paths are from the root directory of the project
DATA_DIR = os.path.join('data', 'SfM')
DATA_IMGS_DIR = os.path.join(DATA_DIR, 'images')
DATA_SPARSE_DIR = os.path.join(DATA_DIR, 'sparse')
DATA_CAMS_FILE_DIR = os.path.join(DATA_SPARSE_DIR, 'cameras.bin')
DATA_IMGS_FILE_DIR = os.path.join(DATA_SPARSE_DIR, 'images.bin')
DATA_POINTS_FILE_DIR = os.path.join(DATA_SPARSE_DIR, 'points3D.bin')