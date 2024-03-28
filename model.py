"""
Tentative definition of the model.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CNNArmController(nn.Module):
    def __init__(self, output_size):
        super(CNNArmController, self).__init__()
        # Use a pre-trained ResNet18 model, remove the top layer
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
        
        # Define the LSTM part
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, output_size)

    def forward(self, x, hidden):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        cnn_out = self.resnet(x)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)
        lstm_out, hidden = self.lstm(cnn_out, hidden)
        out = self.fc(lstm_out[:, -1])
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, 256), torch.zeros(1, batch_size, 256))


# output_size = 6
# Representing the joint angles for each of the 6 servo motors


# The control loop could look something like this
def control_loop(model, camera, servo_controller, frequency):
    period = 1.0 / frequency
    while True:
        start_time = time.time()
        
        # Capture and preprocess the image
        image = camera.capture_image()
        preprocessed_image = preprocess_image(image)
        
        # Run model inference
        predicted_angles = model.predict(preprocessed_image)
        
        # Update servo motor positions
        servo_controller.update_positions(predicted_angles)
        
        # Wait for the remaining time of the control loop period
        elapsed_time = time.time() - start_time
        remaining_time = period - elapsed_time
        if remaining_time > 0:
            time.sleep(remaining_time)