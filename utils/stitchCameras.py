import os
import numpy as np
import shutil
import subprocess
import time
from utils import colmapUtils

def colmapByChunk(
        img_folder=r'C:\Users\JOEL\OneDrive\Documents\Columbia\Fall_23\Deep_Learning\project\original\preprocessed_local_single', 
        workspace=r'C:\Users\JOEL\OneDrive\Documents\Columbia\Fall_23\Deep_Learning\project\sfm\auto', 
        chunk_size=100, 
        overlap_halfsize=25,
        copy_imgs=False,
        rerun_sfm=False,
        colmap=r"C:\Users\JOEL\Downloads\COLMAP-3.8-windows-cuda\COLMAP-3.8-windows-cuda\COLMAP.bat",
        db=None):#r"C:\Users\JOEL\OneDrive\Documents\Columbia\Fall_23\Deep_Learning\project\sfm\final\database.db"):
    
    os.chdir(os.path.join(workspace, 'images'))
    
    if copy_imgs:

        # Grab images from original folder
        images = {int(f.split('_')[1]): f for f in os.listdir(img_folder)}

        # Helper function to switch to a new folder
        splits = list()
        def newFolder(cursor):
            end = str(min(cursor + chunk_size + overlap_halfsize * (1 + bool(cursor)), len(images))).zfill(4)
            folder_name = f'{str(cursor).zfill(4)}-{end}'
            os.mkdir(folder_name)
            dest_folder = os.path.abspath(folder_name)
            splits.append(dest_folder)
            return dest_folder, cursor
        
        # Copy the images over
        dest_folder, cursor = newFolder(0)
        ignore = overlap_halfsize
        while cursor in images.keys():
            shutil.copy(os.path.join(img_folder, images[cursor]), os.path.join(dest_folder, images[cursor]))
            if cursor != ignore and not (cursor - overlap_halfsize) % chunk_size:
                ignore = cursor
                cursor -= (overlap_halfsize * 2)
                dest_folder, cursor = newFolder(cursor)
            cursor += 1

    else:
        splits = [os.path.abspath(f) for f in os.listdir(os.path.join(workspace, 'images'))]

    # Run COLMAP
    os.chdir(os.path.join(workspace, 'models'))
    for i, folder in enumerate(splits):

        # Create the workspace
        ws = os.path.abspath(os.path.basename(folder))
        if not os.path.exists(ws):
            os.mkdir(ws)
        if not rerun_sfm and os.path.exists(os.path.join(ws, 'sparse')):
            print(f'Skipping {os.path.basename(folder)}')
            continue

        # Define commands
        if db:
            cmd = 'mapper'
            args = {'image_path': folder, 'output_path': ws, 'database_path': db, 
                    'Mapper.ba_global_use_pba': 1, 'Mapper.max_num_models': 1}
        else:
            cmd = 'automatic_reconstructor'
            args =  {'image_path': folder, 'workspace_path': ws, 'data_type': 'video', 
                     'quality': 'low', 'single_camera': 1, 'dense': 0}
        argstring = ' '.join([f'--{k} {v}' for k, v in args.items()])

        # Run SfM
        try:
            subprocess.run(f'{colmap} {cmd} {argstring}', shell=True, check=True)
        except:
            time.sleep(5)
            subprocess.run(f'{colmap} {cmd} {argstring}', shell=True, check=True)
        print(f'({i}/{len(splits)})\t Reconstructed {ws}')

    return os.getcwd()


def stitchVectors(v1:np.array, v2:np.array, model_num:str=str())->np.array:

    # Extract overlapping cameras
    left_indices, right_indices = v1[:, 0], v2[:, 0]
    overlap = list(np.intersect1d(left_indices, right_indices).astype(int))
    print(f'\t\t Found {len(overlap)} overlaps')
    if not len(overlap):
        # print(list(sorted(left_indices.astype(int))), '\n', list(sorted(right_indices.astype(int))), '\n\n')
        return v1
    left, right = v1[np.isin(v1[:, 0], overlap), 1:], v2[np.isin(v2[:, 0], overlap), 1:]
    
    # Perform least squares fitting
    x, _, _, _ = np.linalg.lstsq(right, left, rcond=None)

    # print(left_indices.shape, right_indices.shape, v1[:, 1:].shape, x.shape)
    
    # Transform vector
    T = np.hstack([right_indices.reshape(-1, 1), v2[:, 1:] @ x])

    # Average overlapping cameras
    s = np.hstack([np.array(overlap).reshape(-1, 1), (left + (right @ x)) / 2])

    # Concatenate and return data
    left_only = np.setdiff1d(left_indices, overlap).astype(int)
    right_only = np.setdiff1d(right_indices, overlap).astype(int)
    v3 = np.concatenate((v1[np.isin(v1[:, 0], left_only), :], s, T[np.isin(T[:, 0], right_only), :]), axis=0)
    v3 = v3[np.argsort(v3[:, 0])]
    return v3

def stitchCameras(folder=r'C:\Users\JOEL\OneDrive\Documents\Columbia\Fall_23\Deep_Learning\project\sfm\split'): #auto\models

    if folder is None:
        folder = colmapByChunk()

    os.chdir(folder)
    cameras = list()

    for split in sorted(os.listdir(folder)):
        for model_num in os.listdir(os.path.join(split, 'sparse')):

            if model_num.endswith('.bin'):
                continue

            # Extract data
            img_bin = os.path.abspath(os.path.join(split, 'sparse', model_num, 'images.bin'))
            print(f'Reading cameras from {img_bin}')
            imageBase = colmapUtils.read_extrinsics_binary(img_bin)

            # Save images, rotations and translations to lists
            cams = list()

            for key in imageBase.keys():

                # Extract the info
                iBase = imageBase[key]
                picnum = int(iBase.name.split('_')[1])

                # Get the 6DOF vector (rotation and translation)
                # First, extract the rotation in Quaternion form
                w, x, y, z = iBase.qvec

                # Roll (x-axis rotation)
                sinr_cosp = 2.0 * (w * x + y * z)
                cosr_cosp = 1.0 - 2.0 * (x**2 + y**2)
                roll = np.arctan2(sinr_cosp, cosr_cosp)

                # Pitch (y-axis rotation)
                sinp = 2.0 * (w * y - z * x)
                pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

                # Yaw (z-axis rotation)
                siny_cosp = 2.0 * (w * z + x * y)
                cosy_cosp = 1.0 - 2.0 * (y**2 + z**2)
                yaw = np.arctan2(siny_cosp, cosy_cosp)

                # Now, extract the position
                tvec = iBase.tvec

                # Add to lists
                cams.append([picnum, roll, pitch, yaw, *tvec])

            # Now, convert the lists to arrays
            cameras.append(np.array(cams))

    cameras.reverse()

    # Iteratively correct camera positions
    initial_num = len(cameras)
    all_models = [cameras.pop()]
    print(f'({all_models[0].shape[0]})\t Starting with {int(all_models[0][:, 0].min())}-{int(all_models[0][:, 0].max())}')

    while cameras:

        newcam = cameras.pop()
        print(f'({newcam.shape[0]})\t Adding {int(newcam[:, 0].min())}-{int(newcam[:, 0].max())}')

        found = False
        for i, model in enumerate(all_models):

            prev = model.shape[0]
            model = stitchVectors(model, newcam)
            if model.shape[0] > prev:
                all_models[i] = model
                found = True
                break

        if found:
            print(f'({model.shape[0]})\t\t Added cameras')
        else:
            all_models.append(newcam)
            print(f'({model.shape[0]})\t\t Added distinct model')

    print(f'Stitched {initial_num} models into {len(all_models)}: {list(sorted([m.shape[0] for m in all_models]))}')
    return all_models