import os
import h5py
import fnmatch
from postprocess_episodes import load_hdf5
from utils import find_all_hdf5, preprocess_base_action, get_norm_stats, flatten_list
import numpy as np
import cv2

# dataset_dir_l = 'E:\\act-plus-plus-main\\act-plus-plus-main'
dataset_dir_l = './mirror_trans'
hdf5_files = find_all_hdf5(dataset_dir_l, skip_mirrored_data= False)
print(hdf5_files)
norm_stats, _ = get_norm_stats(hdf5_files)

# print(norm_stats)
start_ts = 5
dataset_path = hdf5_files[0]

with h5py.File(dataset_path, 'r') as root:
    try:  # some legacy data does not have this attribute
        is_sim = root.attrs['sim']
    except:
        is_sim = False
    compressed = root.attrs.get('compress', False)
    print(is_sim, compressed)
    if '/base_action' in root:
        base_action = root['/base_action'][()]
        base_action = preprocess_base_action(base_action)
        action = np.concatenate([root['/action'][()], base_action], axis=-1)
    else:
        action = root['/action'][()]
        dummy_base_action = np.zeros([action.shape[0], 2])
        action = np.concatenate([action, dummy_base_action], axis=-1)
#
    original_action_shape = action.shape
    episode_len = original_action_shape[0]
    print(original_action_shape, episode_len)
    # get observation at start_ts only
    qpos = root['/observations/qpos'][start_ts]
    qvel = root['/observations/qvel'][start_ts]
    # image_dict = dict()
    image_dict = dict()
    for cam_name in ['top', 'left_wrist', 'right_wrist']:
        print(root[f'/observations/images/{cam_name}'])
        image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
        print(image_dict[cam_name].shape)
    if compressed:
        for cam_name in image_dict.keys():
            decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
            print(decompressed_image.shape)
            image_dict[cam_name] = np.array(decompressed_image)
            print(image_dict[cam_name].shape)
    # get all actions after and including start_ts
    if is_sim:
        action = action[start_ts:]
        action_len = episode_len - start_ts
    else:
        action = action[max(0, start_ts - 1):]  # hack, to make timesteps more aligned
        action_len = episode_len - max(0, start_ts - 1)  # hack, to make timesteps more aligned
