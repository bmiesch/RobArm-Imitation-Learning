import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from model import ResNet18ForRobotArm
import matplotlib.pyplot as plt
import json

DATASET_PATH_DIAG = "../data/archived/diagonal"
DATASET_PATH_SIDE = "../data/archived/side"
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

        self.flat_data = []
        for iteration, records in self.labels.items():
            iteration_dir = os.path.join(os.path.join(root_dir, IMAGES_FOLDER), f"task_{iteration.split('_')[-1]}")
            if os.path.exists(iteration_dir):
                last_valid_joints = None
                for record in records:
                    if record["joint_angles"] is not None:
                        if last_valid_joints is not None:
                            record["joint_angles"] = [joint if joint is not None else last_valid for joint, last_valid in zip(record["joint_angles"], last_valid_joints)]
                        last_valid_joints = [joint if joint is not None else 0 for joint in record["joint_angles"]]  # Replace None with 0 if it's the first record
                    else:
                        continue

                    # At this point, record["joint_angles"] should have no None values
                    self.flat_data.append({
                        "image_filename": record["image_filename"],
                        "joint_angles": record["joint_angles"]
                    })

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.flat_data[idx]["image_filename"].lstrip('data/'))
        image = Image.open(img_name)
        joints = self.flat_data[idx]["joint_angles"]

        if joints is None or any(joint is None for joint in joints):
            print(f"Warning: Missing joint angles for image {img_name}. Replacing with zeros.")
            joints = [0.0] * 6
        
        joints = torch.tensor(joints, dtype=torch.float)
        if self.transform:
            image = self.transform(image)
        return image, joints

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize datasets
# dataset1 = ArmDataset(json_file=LABELS_FILE, root_dir=DATASET_PATH_DIAG, transform=transform)
# dataset2 = ArmDataset(json_file=LABELS_FILE, root_dir=DATASET_PATH_SIDE, transform=transform)
# full_dataset = ConcatDataset([dataset1, dataset2])
full_dataset = ArmDataset(json_file=LABELS_FILE, root_dir=DATASET_PATH, transform=transform)

# Splitting dataset into training and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ResNet18ForRobotArm(num_output=6).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Setup the plot
fig, ax = plt.subplots()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
train_line, = ax.plot([], [], label='Training Loss')
val_line, = ax.plot([], [], label='Validation Loss')
plt.legend()
plt.show(block=False)

def update_plot(epoch):
    train_line.set_data(range(epoch + 1), TRAIN_LOSSES)
    val_line.set_data(range(epoch + 1), VAL_LOSSES)
    ax.relim()
    ax.autoscale_view(True, True, True)
    plt.draw()
    plt.pause(0.1)

# Training function
def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (images, joints) in enumerate(train_dataloader):
            images, joints = images.to(device), joints.to(device)
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, joints)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 10 == 0:  # Print every 10 batches
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}, Loss: {loss.item()}")

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {epoch_loss}")
        TRAIN_LOSSES.append(epoch_loss)

        # Validation phase
        val_loss = validate(model, val_dataloader, loss_fn)
        VAL_LOSSES.append(val_loss)

        update_plot(epoch)

def validate(model, dataloader, loss_fn):
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, joints in dataloader:
            images, joints = images.to(device), joints.to(device)
            predictions = model(images)
            val_loss = loss_fn(predictions, joints)
            running_val_loss += val_loss.item()
    average_val_loss = running_val_loss / len(dataloader)
    print(f"Validation Loss: {average_val_loss}")
    return average_val_loss

# Execute training
epochs = 15
train(model, train_dataloader, val_dataloader, loss_fn, optimizer, epochs)

# Save the model
torch.save(model.state_dict(), "cnn_arm_controller.pth")

# Save the losses to a file
with open('losses.txt', 'w') as f:
    f.write("epoch,train_loss,val_loss\n")
    for train_loss, val_loss in zip(TRAIN_LOSSES, VAL_LOSSES):
        f.write(f"{train_loss},{val_loss}\n")

plt.show()
input("Press Enter to continue...")