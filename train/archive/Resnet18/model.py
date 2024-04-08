import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet18ForRobotArm(nn.Module):
    def __init__(self, num_output=6):
        super(ResNet18ForRobotArm, self).__init__()
        # Load a pretrained ResNet-18 model
        self.resnet18 = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        num_ftrs = self.resnet18.fc.in_features  # Correctly access the in_features of the last fc layer
        self.resnet18.fc = nn.Identity()  # Replace the last fc layer with an identity mapping
        
        # Add a new fully connected layer for our specific task
        self.fc = nn.Linear(num_ftrs, num_output)

    def forward(self, x):
        # Pass the input through the ResNet-18 model
        x = self.resnet18(x)
        # Pass the output through the new fully connected layer
        x = self.fc(x)
        return x