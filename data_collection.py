import cv2
import numpy as np
import os
import json
import time
import threading
from control import RoboticArm
import rospy


class DataCollection:
    def __init__(self, camera_index, image_path="data/images", signal_path="data/", crop_params=None):
        self.camera = cv2.VideoCapture(camera_index)
        self.image_path = image_path
        self.signal_path = signal_path
        self.crop_params = crop_params
        self.data_buffer = []
        self.robotic_arm = RoboticArm()
        self.running = False

    def save_task_data(self, task_iteration):
        """
        Saves the images and joint angles (data_buffer) for an iteration to the specified path.
        Pretty-prints each data entry on a new line for better readability.
        """
        timestamp = int(time.time() * 1000)
        task_folder = f"{self.image_path}/task_{task_iteration}"
        os.makedirs(task_folder, exist_ok=True)

        # Prepare the data for the current iteration
        current_iteration_data = []
        for i, (image, joints) in enumerate(self.data_buffer):
            image_filename = f"{task_folder}/image_{timestamp}_{i}.png"
            cv2.imwrite(image_filename, image)
            current_iteration_data.append({
                "timestamp": timestamp,
                "image_index": i,
                "image_filename": image_filename,
                "joint_angles": joints
            })

        # Load existing data or initialize if starting fresh
        json_file_path = f"{self.signal_path}/joint_angles.json"
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                data = json.load(f)
        else:
            data = {"data": {}}

        # Append the current iteration's data
        data["data"][f"iteration_{task_iteration}"] = current_iteration_data

        # Write the updated data back to the file
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=4)

        self.data_buffer = []

    def task_data_collection_thread(self):
        """
        Collects images and joint angles from the robotic arm and stores them in the data_buffer.
        """
        self.running = True
        while self.running:
            img = self.capture_image()
            joints_angles = self.robotic_arm.read_joint_angles()
            self.data_buffer.append((img.copy(), joints_angles))
            time.sleep(0.1)

    def start_task_data_collection(self):
        """
        Starts the task data collection thread.
        """
        rospy.loginfo("Starting task data collection")
        self.task_data_collection_thread_instance = threading.Thread(target=self.task_data_collection_thread)
        self.task_data_collection_thread_instance.daemon = True
        self.task_data_collection_thread_instance.start()

    def stop_task_data_collection(self, task_iteration):
        """
        Stops the task data collection thread and saves the data for the specified task iteration.
        """
        rospy.loginfo("Stopping task data collection")
        self.running = False
        if self.task_data_collection_thread_instance:
            self.task_data_collection_thread_instance.join()
        if task_iteration != -1:
            self.save_task_data(task_iteration)

    def capture_image(self):
        """
        Captures an image from the camera and returns it.
        """
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to grab frame")
            return None
        resized_frame = cv2.resize(frame, (548, 274), interpolation=cv2.INTER_AREA)
        if self.crop_params is not None:
            resized_frame = resized_frame[self.crop_params[1]:self.crop_params[1] + self.crop_params[3],
                                          self.crop_params[0]:self.crop_params[0] + self.crop_params[2]]
        return resized_frame

    def calibrate_camera(self):
        """
        Calibrates the camera by selecting a region of interest (ROI) to crop from the image.
        """
        print("Calibrating camera. Press 'q' to quit, 'c' to crop.")

        while True:
            frame = self.capture_image()
            cv2.imshow('Calibration', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                cv2.destroyAllWindows()
                self.crop_params = self.select_fixed_size_roi(frame)
                if self.crop_params is not None:
                    print(f"Crop selected at: {self.crop_params}")
                    break

        cv2.destroyAllWindows()

    def select_fixed_size_roi(self, frame, size=(240, 240)):
        """
        Selects a fixed 240x240 region of interest (ROI) to crop from the image.
        ResNet18 requires a 240x240 input image for training.
        """
        roi_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        def mouse_event(event, x, y, flags, param):
            nonlocal roi_center
            if event == cv2.EVENT_MOUSEMOVE:
                roi_center = (x, y)
                clone = frame.copy()
                cv2.rectangle(clone, (x - size[0] // 2, y - size[1] // 2),
                              (x + size[0] // 2, y + size[1] // 2), (0, 255, 0), 2)
                cv2.imshow("Calibration", clone)

        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_event)

        print("Move the box to the desired location and press 'c' to confirm, 'q' to quit.")

        crop_params = None
        while True:
            clone = frame.copy()
            cv2.rectangle(clone, (roi_center[0] - size[0] // 2, roi_center[1] - size[1] // 2),
                          (roi_center[0] + size[0] // 2, roi_center[1] + size[1] // 2), (0, 255, 0), 2)
            cv2.imshow("Calibration", clone)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                x, y = roi_center
                crop_params = (x - size[0] // 2, y - size[1] // 2, size[0], size[1])
                break
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
        return crop_params
