import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from model import ResNet18LSTMForRobotArm
import matplotlib.pyplot as plt
import json

DATASET_PATH = "../data"
IMAGES_FOLDER = "images"
LABELS_FILE = "joint_angles.json"

VAL_LOSSES = []
TRAIN_LOSSES = []

class ArmSequenceDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None, sequence_length=60):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            sequence_length (int): Fixed length of sequences. Shorter sequences will be padded, longer will be truncated.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        with open(os.path.join(root_dir, json_file), 'r') as f:
            self.labels = json.load(f)['data']

        self.sequences = []
        self.joint_sequences = []
        for iteration, records in self.labels.items():
            iteration_dir = os.path.join(root_dir, "images", f"task_{iteration.split('_')[-1]}")
            if os.path.exists(iteration_dir):
                image_sequence = []
                joint_sequence = []
                last_valid_joints = None
                for record in records:
                    if record["joint_angles"] is not None:
                        if last_valid_joints is not None:
                            record["joint_angles"] = [joint if joint is not None else last_valid for joint, last_valid in zip(record["joint_angles"], last_valid_joints)]
                        last_valid_joints = [joint if joint is not None else 0 for joint in record["joint_angles"]]  # Replace None with 0 if it's the first record
                
                        image_path = record["image_filename"]
                        image_sequence.append(image_path)
                        joint_sequence.append(record["joint_angles"])
                
                # Ensure the sequence is of fixed length
                if len(image_sequence) > self.sequence_length:
                    # Truncate the sequence if it's too long
                    image_sequence = image_sequence[:self.sequence_length]
                    joint_sequence = joint_sequence[:self.sequence_length]
                elif len(image_sequence) < self.sequence_length:
                    # Pad the sequence if it's too short
                    padding_length = self.sequence_length - len(image_sequence)
                    image_sequence.extend([None] * padding_length)
                    joint_sequence.extend([[0.0] * 6] * padding_length)
                
                self.sequences.append(image_sequence)
                self.joint_sequences.append(joint_sequence)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_sequence = self.sequences[idx]
        joint_sequence = self.joint_sequences[idx]

        images = []
        for img_path in image_sequence:
            if img_path is not None:
                img_path = os.path.join("..", img_path)
                image = Image.open(img_path)
                if self.transform:
                    image = self.transform(image)
            else:
                # Create a dummy image for padding
                image = torch.zeros((3, 224, 224))
            images.append(image)

        images = torch.stack(images)  # Shape: (sequence_length, C, H, W)
        joints = torch.tensor(joint_sequence, dtype=torch.float)  # Shape: (sequence_length, num_joints)

        return images, joints

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize datasets
full_dataset = ArmSequenceDataset(json_file=LABELS_FILE, root_dir=DATASET_PATH, transform=transform)

# Splitting dataset into training and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
print("After dataloaders")

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ResNet18LSTMForRobotArm().to(device)
print("model loaded")
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
            print(f'Starting sequence {i}')
            images = images.to(device)
            joints = joints.to(device)
            print("Loaded batch")

            optimizer.zero_grad()
            predictions = model(images)
            print("Predictions")
            loss = loss_fn(predictions, joints)
            print("Loss")
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        TRAIN_LOSSES.append(epoch_loss)

        # Validation phase
        val_loss = validate(model, val_dataloader, loss_fn)
        VAL_LOSSES.append(val_loss)

        update_plot(epoch)
        print(f"Epoch {epoch+1}, Training Loss: {epoch_loss}, Validation Loss: {val_loss}")

def validate(model, dataloader, loss_fn):
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, joints in dataloader:
            images = images.to(device)
            joints = joints.to(device)
            
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
torch.save(model.state_dict(), "resnet18_lstm_arm_controller.pth")

# Save the losses to a file
with open('losses.txt', 'w') as f:
    f.write("epoch,train_loss,val_loss\n")
    for epoch, (train_loss, val_loss) in enumerate(zip(TRAIN_LOSSES, VAL_LOSSES)):
        f.write(f"{epoch+1},{train_loss},{val_loss}\n")

plt.show()
input("Press Enter to continue...")
