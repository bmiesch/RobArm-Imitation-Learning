import cv2
import torch
import numpy as np
from train.model import ResNet18ForRobotArm 
from control import RoboticArm
from torchvision import transforms
from PIL import Image

class RobotController:
    def __init__(self, camera_index, model_path):
        self.robotic_arm = RoboticArm()
        self.camera = cv2.VideoCapture(camera_index)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path).to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_path):
        model = ResNet18ForRobotArm(num_output=6)  # Adjust this based on your model
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model

    def capture_image(self):
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to grab frame")
            return None
        cv2.imshow("Live Video Stream", frame)
        cv2.waitKey(1)
        return frame

    def preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image).unsqueeze(0).to(self.device)  # Add batch dimension and move to GPU
        return image

    def predict_joint_angles(self, image):
        with torch.no_grad():
            image = self.preprocess_image(image)
            predictions = self.model(image)
            joint_angles = predictions.squeeze().cpu().tolist()  # Move data to CPU before converting to Python list
        return joint_angles

    def control_loop(self):
        while True:
            image = self.capture_image()
            if image is None:
                continue
            joint_angles = self.predict_joint_angles(image)
            print(joint_angles)
            # Uncomment and ensure safety before enabling movement
            # self.robotic_arm.set_custom_position(joint_angles)

if __name__ == "__main__":
    controller = RobotController(camera_index=1, model_path='train/cnn_arm_controller.pth')
    controller.control_loop()