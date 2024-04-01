import cv2 as cv
import torch
import numpy as np
from train.model import ResNet18ForRobotArm 
from control import RoboticArm
from torchvision import transforms
from PIL import Image

class RobotController:
    def __init__(self, camera_index, model_path):
        self.robotic_arm = RoboticArm()
        self.camera = cv.VideoCapture(camera_index)
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
    
    def calibrate_camera(self):
        print("Calibrating camera. Press 'c' to crop and 'q' to quit.")
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame")
                continue
            resized_frame = cv.resize(frame, (548, 274), interpolation=cv.INTER_AREA)
            cv.imshow('Calibration', resized_frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                # Assuming select_fixed_size_roi returns the crop parameters
                self.crop_params = self.select_fixed_size_roi(resized_frame)
                print(f"Crop selected at: {self.crop_params}")
                break
            elif key == ord('q'):
                break
        cv.destroyAllWindows()
    
    def select_fixed_size_roi(self, frame, size=(240, 240)):
        roi_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        def mouse_event(event, x, y, flags, param):
            nonlocal roi_center
            if event == cv.EVENT_MOUSEMOVE:
                roi_center = (x, y)
                clone = frame.copy()
                cv.rectangle(clone, (x - size[0] // 2, y - size[1] // 2),
                            (x + size[0] // 2, y + size[1] // 2), (0, 255, 0), 2)
                cv.imshow("Calibration", clone)

        cv.namedWindow("Calibration")
        cv.setMouseCallback("Calibration", mouse_event)

        print("Move the box to the desired location and press 'c' to confirm, 'q' to quit.")

        crop_params = None
        while True:
            key = cv.waitKey(1) & 0xFF
            if key == ord('c'):
                x, y = roi_center
                crop_params = (x - size[0] // 2, y - size[1] // 2, size[0], size[1])
                break
            elif key == ord('q'):
                break

        cv.destroyAllWindows()
        return crop_params

    def capture_image(self):
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to grab frame")
            return None
        resized_frame = cv.resize(frame, (548, 274), interpolation=cv.INTER_AREA)
        if self.crop_params is not None:
            x, y, w, h = self.crop_params
            cropped_frame = resized_frame[y:y+h, x:x+w]
        else:
            raise ValueError("Crop parameters not set")
        return cropped_frame

    def preprocess_image(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
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
            
            # Manual confirmation
            # input("Press Enter to move to the next position, or Ctrl+C to exit...")
            
            self.robotic_arm.set_custom_position(joint_angles)

if __name__ == "__main__":
    controller = RobotController(camera_index=1, model_path='train/cnn_arm_controller.pth')
    controller.calibrate_camera()
    controller.control_loop()