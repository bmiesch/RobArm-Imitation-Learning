import h5py
import numpy as np
import os
import json
from PIL import Image
import torch
from training.utils import get_norm_stats

print(torch.__version__)
print(torch.cuda.is_available())


def load_image(image_path):
    """Load an image from the specified path and return as a numpy array."""
    with Image.open(image_path) as img:
        return np.array(img)

def degrees_to_radians(degrees):
    radians = degrees * (np.pi / 180)
    radians = (radians + np.pi) % (2 * np.pi) - np.pi
    return radians

def radians_to_degrees(radians):
    degrees = radians * (180 / np.pi)
    degrees = (degrees + 360) % 360
    return degrees

def handle_nan_values(current_values, last_valid_values):
    """Replace NaN values in the current data array with the last valid values."""
    if last_valid_values is None:
        last_valid_values = current_values

    is_nan = np.isnan(current_values)
    current_values[is_nan] = last_valid_values[is_nan]
    return current_values

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
                
                last_valid_qpos = None
                last_valid_action = None

                for i, record in enumerate(records):
                    if i >= MAX_FRAMES:
                        break

                    current_qpos = np.array(record['joint_angles'], dtype=np.float32)
                    current_action = np.array(record['goal_joints'], dtype=np.float32)

                    # Handle null values
                    current_qpos = handle_nan_values(current_qpos, last_valid_qpos)
                    current_action = handle_nan_values(current_action, last_valid_action)

                    # Update last valid records
                    last_valid_qpos = current_qpos
                    last_valid_action = current_action

                    qpos_dataset[i] = degrees_to_radians(current_qpos)
                    qvel_dataset[i] = np.zeros(len(record['joint_angles']), dtype=np.float32)  # Placeholder if no velocity data
                    action_dataset[i] = degrees_to_radians(current_action)
                    
                    image_filename = record['image_filename'].split('/')[-1]
                    image_path = os.path.join(image_base_dir, f"task_{episode_id}", image_filename)
                    image_data = load_image(image_path)
                    image_list.append(image_data)

                # Save the truncated list of images
                image_dataset = images_group.create_dataset(camera_name, data=np.array(image_list), dtype='uint8')
            
            episode_counter += 1


# Example usage
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'data')
image_base_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data/bowl_data/diagonal/images')
print(image_base_dir)

# Load JSON data
with open('data_example.json', 'r') as json_file:
    data = json.load(json_file)

create_hdf5_from_json(data, dataset_dir, image_base_dir, task_name="diagonal")

from training.utils import EpisodicDataset

task_name = "diagonal"
print(dataset_dir)
task_dir = os.path.join(dataset_dir, task_name)
camera_names = ["camera1"]  # This should match the name used in the HDF5 file
print(task_dir)
norm_stats = get_norm_stats(task_dir, len(os.listdir(task_dir)))

dataset = EpisodicDataset([i for i in range(len(os.listdir(task_dir)))], os.path.join(dataset_dir, task_name), camera_names, norm_stats)

# Do some operations on the dataset to test that it is working
# for batch_idx, data in enumerate(dataset):
#     print(f"Batch {batch_idx}:")
#     print(data)
#     print()
