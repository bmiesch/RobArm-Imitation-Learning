import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter


class SingleCameraCNNMLP(nn.Module):
    def __init__(self, state_dim):
        super(SingleCameraCNNMLP, self).__init__()
        # Define the backbone model for feature extraction
        self.backbone = self._create_backbone_model('resnet18', pretrained=True, return_interm_layers=False)
        num_channels = 512  # For ResNet18 and ResNet34
        
        # Define the projection layers to downsample the feature map
        self.down_proj = nn.Sequential(
            nn.Conv2d(num_channels, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # This will pool the feature map to size (1, 1)
        )
        
        # Define the MLP for predicting the joint angles
        self.mlp = self._create_mlp(32 + state_dim, 1024, state_dim, hidden_depth=2)

    def _create_backbone_model(self, name, pretrained=True, return_interm_layers=False):
        backbone = getattr(models, name)(pretrained=pretrained)
        # Remove the last fully connected layer
        backbone.fc = nn.Identity()
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        return IntermediateLayerGetter(backbone, return_layers=return_layers)

    def _create_mlp(self, input_dim, hidden_dim, output_dim, hidden_depth):
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*mods)

    def forward(self, image, qpos):
        """
        image: Tensor of shape (batch, channel, height, width)
        qpos: Tensor of shape (batch, qpos_dim)
        """
        features = self.backbone(image)['0']

        features = self.down_proj(features)
        features = features.view(features.size(0), -1)  # Flatten the features
        
        # Concatenate qpos with flattened features
        combined_features = torch.cat([features, qpos], dim=1)
        
        # Predict the joint angles
        joint_angles = self.mlp(combined_features)
        return joint_angles
