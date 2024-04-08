import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18LSTMForRobotArm(nn.Module):
    def __init__(self, num_output=6, hidden_dim=256, num_layers=1, bidirectional=False):
        super(ResNet18LSTMForRobotArm, self).__init__()
        # Load a pretrained ResNet-18 model
        self.resnet18 = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()  # Replace the last fc layer with an identity mapping
        
        # LSTM layer
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        # The input dimension to the LSTM is the number of features from ResNet-18
        self.lstm = nn.LSTM(input_size=num_ftrs, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        
        # Fully connected layer
        # If the LSTM is bidirectional, it doubles the output features
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_output)

    def forward(self, x):
        # x is expected to be of shape (batch, sequence, C, H, W)
        batch_size, sequence_length, C, H, W = x.size()
        # Reshape x to treat the sequence of images as a batch of images
        x = x.view(batch_size * sequence_length, C, H, W)
        # Pass the input through the ResNet-18 model
        x = self.resnet18(x)
        # Reshape x back to (batch, sequence, feature_size)
        x = x.view(batch_size, sequence_length, -1)
        
        # Pass the output through the LSTM layer
        lstm_out, _ = self.lstm(x)
        # Only take the output from the final time step
        # x = lstm_out[:, -1, :]
        x = self.fc(lstm_out)  # Apply the fully connected layer to every timestep
        
        return x