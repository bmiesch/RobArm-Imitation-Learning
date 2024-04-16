import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms
from PIL import Image
from model import SingleCameraCNNMLP
import matplotlib.pyplot as plt

SIDE_DATASET_PATH = "../data/bowl_data/side"
FRONT_DATASET_PATH = "../data/bowl_data/front"
DIAGONAL_DATASET_PATH = "../data/bowl_data/diagonal"
DATASET_PATH = "../data"
IMAGES_FOLDER = "images"
LABELS_FILE = "joint_angles.json"

VAL_LOSSES = []
TRAIN_LOSSES = []

class ArmDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        with open(os.path.join(root_dir, json_file), 'r') as f:
            self.labels = json.load(f)['data']

        self.ordered_data = []
        # Ensure tasks are processed in order by sorting the keys
        for iteration in sorted(self.labels.keys(), key=lambda x: int(x.split('_')[-1])):
            iteration_dir = os.path.join(os.path.join(root_dir, IMAGES_FOLDER), f"task_{iteration.split('_')[-1]}")
            if os.path.exists(iteration_dir):
                records = self.labels[iteration]
                last_valid_joints = None
                for index, record in enumerate(records):
                    if record["joint_angles"] is not None:
                        if last_valid_joints is not None:
                            record["joint_angles"] = [joint if joint is not None else last_valid for joint, last_valid in zip(record["joint_angles"], last_valid_joints)]
                        last_valid_joints = [joint if joint is not None else 0 for joint in record["joint_angles"]]  # Replace None with 0 if it's the first record
                    else:
                        print(f"Warning: Missing joint angles for image {record['image_filename']}")
                        continue

                    if index % 2 == 0:
                        self.ordered_data.append((record["image_filename"], record["joint_angles"]))

    def __len__(self):
        return len(self.ordered_data)

    def __getitem__(self, idx):
        img_path, joints = self.ordered_data[idx]
        # data/images/task_1/image_1712539867982_0.png
        # strip the data/ off of this
        img_path = img_path[5:]
        img_path = os.path.join(self.root_dir, img_path)

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        joints = torch.tensor(joints, dtype=torch.float)
        return image, joints

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize datasets
# full_dataset = ArmDataset(json_file=LABELS_FILE, root_dir=DATASET_PATH, transform=transform)
front_dataset = ArmDataset(json_file=LABELS_FILE, root_dir=FRONT_DATASET_PATH, transform=transform)
side_dataset = ArmDataset(json_file=LABELS_FILE, root_dir=SIDE_DATASET_PATH, transform=transform)
diagonal_dataset = ArmDataset(json_file=LABELS_FILE, root_dir=DIAGONAL_DATASET_PATH, transform=transform)

# full_dataset = ConcatDataset([front_dataset, side_dataset, diagonal_dataset])

# # Since we need to maintain order, we won't use random_split
# num_examples = len(full_dataset)
# print(num_examples)
# train_size = int(0.8 * num_examples)
# val_size = num_examples - train_size

# # Generate indices: Since the dataset is already in order, simply divide it
# indices = list(range(num_examples))
# train_indices = indices[:train_size]
# val_indices = indices[train_size:]

# # Create training and validation subsets
# train_dataset = Subset(full_dataset, train_indices)
# val_dataset = Subset(full_dataset, val_indices)
def split_dataset_by_iteration(dataset, train_ratio=0.8):
    """Splits a dataset by iteration into training and validation sets."""
    num_iterations = len(dataset) // 59  # Assuming each iteration has 60 datapoints
    num_train_iterations = int(num_iterations * train_ratio)
    
    # Calculate the split index
    split_idx = num_train_iterations * 59  # Number of datapoints in the training set
    
    train_dataset = Subset(dataset, range(split_idx))
    val_dataset = Subset(dataset, range(split_idx, len(dataset)))
    
    return train_dataset, val_dataset

# Split each angle dataset
train_side_dataset, val_side_dataset = split_dataset_by_iteration(side_dataset)
print(len(train_side_dataset))
print(len(val_side_dataset))
train_front_dataset, val_front_dataset = split_dataset_by_iteration(front_dataset)
print(len(train_front_dataset))
print(len(val_front_dataset))
train_diagonal_dataset, val_diagonal_dataset = split_dataset_by_iteration(diagonal_dataset)
print(len(train_diagonal_dataset))
print(len(val_diagonal_dataset))

# Concatenate the training and validation datasets across angles
train_dataset = ConcatDataset([train_side_dataset, train_front_dataset, train_diagonal_dataset])
val_dataset = ConcatDataset([val_side_dataset, val_front_dataset, val_diagonal_dataset])


train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=30, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = SingleCameraCNNMLP(state_dim=6).to(device)  # Assuming 6 joint angles to predict
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

checkpoint_path = 'model_checkpoint.pth'

# Load the model from the checkpoint if it exists
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Checkpoint loaded successfully.")
else:
    print("No checkpoint found. Starting from scratch.")


# Training function
def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, joints in train_dataloader:
            images, joints = images.to(device), joints.to(device)
            optimizer.zero_grad()
            predictions = model(images, joints)
            loss = loss_fn(predictions, joints)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_train_loss = running_loss / len(train_dataloader)
        TRAIN_LOSSES.append(average_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, joints in val_dataloader:
                images, joints = images.to(device), joints.to(device)
                predictions = model(images, joints)
                val_loss = loss_fn(predictions, joints)
                running_val_loss += val_loss.item()

        average_val_loss = running_val_loss / len(val_dataloader)
        VAL_LOSSES.append(average_val_loss)

        # Step the scheduler on each epoch
        scheduler.step(val_loss)

        # print(f"Epoch {epoch+1}/{epochs}, Training Loss: {running_loss}, Validation Loss: {running_val_loss}")
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {average_train_loss:.4f}, Validation Loss: {average_val_loss:.4f}")


# Execute training
epochs = 10
train(model, train_dataloader, val_dataloader, loss_fn, optimizer, scheduler, epochs)

# Save the model
torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'model_checkpoint.pth')

# Save the losses to a file
with open('losses.txt', 'a') as f:
    f.write("epoch,train_loss,val_loss\n")
    for epoch, (train_loss, val_loss) in enumerate(zip(TRAIN_LOSSES, VAL_LOSSES)):
        f.write(f"{epoch},{train_loss},{val_loss}\n")
