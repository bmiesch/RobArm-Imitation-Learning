import h5py
import numpy as np
import os
import json
from PIL import Image
import torch
from training.utils import get_norm_stats


def load_image(image_path):
    """Load an image from the specified path and return as a numpy array."""
    with Image.open(image_path) as img:
        return np.array(img)

MAX_FRAMES = 58

def create_hdf5_from_json(data, dataset_dir, image_base_dir, task_name):
    task_dir = os.path.join(dataset_dir, task_name)
    # Ensure the directory exists
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    camera_name = "camera1"
    episode_counter = 0
    
    for iteration, records in data['data'].items():
        episode_id = iteration.split('_')[-1]  # Extract episode id from iteration_xx

        # Check if the episode directory exists 
        # This is because some episode were removed from the dataset
        if os.path.exists(os.path.join(image_base_dir, f"task_{episode_id}")):
            hdf5_filename = os.path.join(task_dir, f'episode_{episode_counter}.hdf5')
            with h5py.File(hdf5_filename, 'w') as hdf5_file:
                hdf5_file.attrs['sim'] = True  # Assuming all data is from simulation as per your setup
                
                # Create groups and datasets
                obs_group = hdf5_file.create_group('observations')
                images_group = obs_group.create_group('images')

                image_list = []
                
                # Initialize datasets with a maximum frame limit
                qpos_dataset = obs_group.create_dataset('qpos', (min(len(records), MAX_FRAMES), len(records[0]['joint_angles'])), dtype='f')
                qvel_dataset = obs_group.create_dataset('qvel', (min(len(records), MAX_FRAMES), len(records[0]['joint_angles'])), dtype='f')
                action_dataset = hdf5_file.create_dataset('action', (min(len(records), MAX_FRAMES), len(records[0]['goal_joints'])), dtype='f')
                
                for i, record in enumerate(records):
                    if i >= MAX_FRAMES:
                        break
                    qpos_dataset[i] = record['joint_angles']
                    qvel_dataset[i] = [0]*len(record['joint_angles'])  # Placeholder if no velocity data
                    action_dataset[i] = record['goal_joints']
                    
                    image_filename = record['image_filename'].split('/')[-1]
                    image_path = os.path.join(image_base_dir, f"task_{episode_id}", image_filename)
                    image_data = load_image(image_path)
                    image_list.append(image_data)

                # Save the truncated list of images
                image_dataset = images_group.create_dataset(camera_name, data=np.array(image_list), dtype='uint8')
            
            episode_counter += 1


# def get_norm_stats(dataset_dir, episode_ids):
#     all_qpos_data = []
#     all_action_data = []
#     for episode_idx in episode_ids:
#         dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
#         with h5py.File(dataset_path, 'r') as root:
#             qpos = root['/observations/qpos'][()]
#             qvel = root['/observations/qvel'][()]
#             action = root['/action'][()]
#         all_qpos_data.append(torch.from_numpy(qpos))
#         all_action_data.append(torch.from_numpy(action))

#     all_qpos_data = torch.stack(all_qpos_data)
#     all_action_data = torch.stack(all_action_data)
#     all_action_data = all_action_data

#     # normalize action data
#     action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
#     action_std = all_action_data.std(dim=[0, 1], keepdim=True)
#     action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

#     # normalize qpos data
#     qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
#     qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
#     qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

#     stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
#              "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
#              "example_qpos": qpos}

#     return stats


# Example usage
dataset_dir = '/Users/bmiesch/Code/ACT-main/data'
image_base_dir = '/Users/bmiesch/Code/ACT-main/diagonal/images'

# Load JSON data
with open('data_example.json', 'r') as json_file:
    data = json.load(json_file)

create_hdf5_from_json(data, dataset_dir, image_base_dir, task_name="diagonal")

from training.utils import EpisodicDataset

task_name = "diagonal"
task_dir = os.path.join(dataset_dir, task_name)
camera_names = ["camera1"]  # This should match the name used in the HDF5 file
norm_stats = get_norm_stats(task_dir, len(os.listdir(task_dir)))

dataset = EpisodicDataset([i for i in range(len(os.listdir(task_dir)))], os.path.join(dataset_dir, task_name), camera_names, norm_stats)

# Do some operations on the dataset to test that it is working
# for batch_idx, data in enumerate(dataset):
#     print(f"Batch {batch_idx}:")
#     print(data)
#     print()
