import h5py
from PIL import Image
import io

def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as file:
        print("Keys in the HDF5 file:")
        print(list(file.keys()))
        for key in file.keys():
            if isinstance(file[key], h5py.Group):
                print(f"Contents of {key}:")
                print(f"Subgroups/Datasets in {key}: {list(file[key].keys())}")
                for subkey in file[key].keys():
                    item = file[key][subkey]
                    if isinstance(item, h5py.Dataset):
                        # Check if the dataset is an image
                        if item.shape[-1] in (3, 4):  # Assuming images are RGB or RGBA
                            print(f"Reading image {subkey}: shape={item.shape}, dtype={item.dtype}")
                            image_data = Image.open(io.BytesIO(item[:]))
                            image_data.show()  # This will display the image
                        else:
                            print(f"Contents of {subkey}: shape={item.shape}, dtype={item.dtype}")
                            print(item[:])  # Print data for non-image datasets
            elif isinstance(file[key], h5py.Dataset):
                data = file[key][:]
                print(f"Contents of {key}: {data}")

# Example usage
read_hdf5('/Users/bmiesch/Code/ACT-main/temp/episode_2.hdf5')

# Assuming you have the path to your HDF5 file and the episode ID
dataset_path = '/Users/bmiesch/Code/ACT-main/temp/episode_2.hdf5'
with h5py.File(dataset_path, 'r') as root:
    camera_group = root['/observations/camera1/images']
    print("Datasets within the camera1 group:")
    for name in camera_group:
        print(name)