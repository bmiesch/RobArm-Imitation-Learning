from config.config import POLICY_CONFIG, TRAIN_CONFIG, TASK_CONFIG
from training.utils import *

import cv2 as cv
import torch
import numpy as np
import pickle
import sys
import os

script_path = os.path.abspath(__file__)
print("Script absolute path:", script_path)
current_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
print("grandparent_dir:", grandparent_dir)
sys.path.append(grandparent_dir)

from control import RoboticArm
from torchvision import transforms
from PIL import Image
import time

policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
cfg = TASK_CONFIG

def degrees_to_radians(degrees):
    """Convert degrees to radians without normalization."""
    return degrees * (np.pi / 180)

def radians_to_degrees(radians):
    """Convert radians to degrees without normalization."""
    return radians * (180 / np.pi)

class RobotController:
    def __init__(self, camera_index, checkpoint_path):
        self.robotic_arm = RoboticArm()
        self.camera = cv.VideoCapture(camera_index)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the policy model as done in evaluate.py
        self.policy = make_policy(POLICY_CONFIG['policy_class'], POLICY_CONFIG)
        model_path = os.path.join(checkpoint_path, 'policy_last.ckpt')
        loading_status = self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
        print(loading_status)
        self.policy.to(self.device)
        self.policy.eval()
        print(f"Model loaded with status: {loading_status}")

        stats_path = os.path.join(checkpoint_path, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        self.pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']
        self.post_process = lambda a: a * self.stats['action_std'] + self.stats['action_mean']

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
        # image = Image.fromarray(image)
        image = cv.resize(image, (240, 240), interpolation=cv.INTER_AREA)
        return image

    def predict_joint_angles(self, image, qpos):
        with torch.no_grad():
            qpos_numpy = np.array(qpos)
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().to(self.device).unsqueeze(0)
            processed_image = {'camera1': self.preprocess_image(image)}
            image = get_image(processed_image, ['camera1'], self.device)

            predictions = self.policy(qpos, image)
            joint_angles = predictions.squeeze().cpu().tolist()

        return joint_angles

    def control_loop(self):
        # Set to above block position
        self.robotic_arm.set_custom_position([90, 80, 50, 50, 265, 135])
            # Initialize buffers for temporal aggregation
        query_frequency = policy_config['num_queries']
        if policy_config['temporal_agg']:
            query_frequency = 1
            num_queries = policy_config['num_queries']
        
        if policy_config['temporal_agg']:
            all_time_actions = torch.zeros([cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]).to(device)
        # qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)

        with torch.inference_mode():
            # Main control loop
            for t in range(cfg['episode_len']):
                image = self.capture_image()
                processed_image = {'camera1': self.preprocess_image(image)}
                image = get_image(processed_image, ['camera1'], self.device)

                qpos = self.robotic_arm.read_joint_angles()
                while any(joint is None for joint in qpos):
                    time.sleep(0.1)
                    qpos = self.robotic_arm.read_joint_angles()
                qpos_radians = [degrees_to_radians(joint) for joint in qpos]
                qpos_numpy = np.array(qpos_radians)
                qpos = self.pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(self.device).unsqueeze(0)


                # Query the policy
                if t % query_frequency == 0:
                    all_actions = self.policy(qpos, image)

                # Handle temporal aggregation
                if policy_config['temporal_agg']:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    print("index: ", t % query_frequency)
                    raw_action = all_actions[:, t % query_frequency]

                # Post-process and execute actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = self.post_process(raw_action)
                joint_angles_degrees = [radians_to_degrees(joint) for joint in action]
                print(joint_angles_degrees)

                # Manual confirmation
                # input("Press Enter to move to the next position, or Ctrl+C to exit...")
                
                self.robotic_arm.set_custom_position(joint_angles_degrees)
                time.sleep(0.01)
    
    def control_loop_continuous(self):
        # Set to initial position
        self.robotic_arm.set_custom_position([90, 80, 50, 50, 265, 135])

        # Initialize buffers for temporal aggregation if needed
        query_frequency = policy_config['num_queries']
        if policy_config['temporal_agg']:
            query_frequency = 1
            num_queries = policy_config['num_queries']
            all_time_actions = torch.zeros([num_queries, cfg['state_dim']]).to(self.device)

        try:
            while True:  # Continuous loop
                with torch.inference_mode():
                    image = self.capture_image()
                    processed_image = {'camera1': self.preprocess_image(image)}
                    image = get_image(processed_image, ['camera1'], self.device)

                    qpos = self.robotic_arm.read_joint_angles()
                    while any(joint is None for joint in qpos):
                        time.sleep(0.1)
                        qpos = self.robotic_arm.read_joint_angles()
                    qpos_radians = [degrees_to_radians(joint) for joint in qpos]
                    qpos_numpy = np.array(qpos_radians)
                    qpos = self.pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().to(self.device).unsqueeze(0)

                    # Query the policy for the next action
                    raw_action = self.policy(qpos, image)

                    # Post-process and execute actions
                    action = raw_action[:, 0]
                    action = action.squeeze(0).cpu().numpy()
                    action = self.post_process(action)
                    joint_angles_degrees = [radians_to_degrees(joint) for joint in action]
                    print(joint_angles_degrees)

                    self.robotic_arm.set_custom_position(joint_angles_degrees)
                    time.sleep(0.01)  # Small delay to prevent overwhelming the robot controller

        except KeyboardInterrupt:
            print("Manual interruption received. Stopping the control loop.")

if __name__ == "__main__":
    controller = RobotController(camera_index=1, checkpoint_path="checkpoints/diagonal")
    controller.calibrate_camera()
    controller.control_loop()
