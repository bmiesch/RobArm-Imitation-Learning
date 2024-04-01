import rospy
import Arm_Lib
import cv2 as cv
import numpy as np
from time import sleep
from dofbot_info.srv import kinemarics, kinemaricsRequest, kinemaricsResponse
from control import RoboticArm
from external.dofbot_config import Arm_Calibration

class PickAndPlace:
    def __init__(self):
        self.robotic_arm = RoboticArm()
        self.n = rospy.init_node('dofbot_pickup', anonymous=True)
        self.client = rospy.ServiceProxy("dofbot_kinemarics", kinemarics)

        # Calibrate
        self.calibration = Arm_Calibration()
        self.calibrate_camera()
    
    def calibrate_camera(self):
        capture = cv.VideoCapture(0)

        if not capture.isOpened():
            rospy.loginfo("Failed to open the camera")
            return
        
        _, img = capture.read()

        cur_angles = self.robotic_arm.read_joint_angles()
        threshold = 140

        self.dp, img = self.calibration.calibration_map(img, cur_angles, threshold)

        # Save the joint angles and threshold value to a file
        # try:
        #     write_XYT(XYT_path, joint_angles, threshold_value)
        # except Exception:
        #     print("Failed to write XYT_config")

        # Close the camera
        capture.release()


    def get_block_position(self, image, color_hsv):
        HSV_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        img = cv.inRange(HSV_img, color_hsv[0], color_hsv[1])
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 1000:
                x, y, w, h = cv.boundingRect(cnt)
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                point_x = float(x + w / 2)
                point_y = float(y + h / 2)
                return (round(((point_x - 320) / 4000), 5), round(((480 - point_y) / 3000) * 0.8+0.19, 5))
        
        return None

    def calculate_inverse_kinematics(self, posxy):
        self.client.wait_for_service()
        request = kinemaricsRequest()
        request.tar_x = posxy[0]
        request.tar_y = posxy[1]
        request.kin_name = "ik"
        try:
            response = self.client.call(request)
            if isinstance(response, kinemaricsResponse):
                joints = [0.0, 0.0, 0.0, 0.0, 0.0]
                joints[0] = response.joint1
                joints[1] = response.joint2
                joints[2] = response.joint3
                joints[3] = response.joint4
                joints[4] = response.joint5
                if joints[2] < 0:
                    joints[1] += joints[2] * 3 / 5
                    joints[3] += joints[2] * 3 / 5
                    joints[2] = 0
                return joints
        except Exception:
            rospy.loginfo("arg error")

    def move_block(self, joints):
        rospy.loginfo(f"Target Location: {joints}")
        block_joints = [joints[0], joints[1], joints[2], joints[3], 265, 30]
        joints_uu = [90, 80, 50, 50, 265, 135]
        joints_up = [joints[0], 80, 50, 50, 265, 30]

        # Move over the block's position
        self.robotic_arm.move_servos(joints_uu, 1000)
        rospy.loginfo("Moved over blocks position")
        sleep(1)

        # Move to block position
        self.robotic_arm.move_servos(block_joints, 500)
        rospy.loginfo("Moved to block position")
        sleep(0.5)

        # Grasp and clamp the block
        self.robotic_arm.move_single_servo(6, 135, 500)
        rospy.loginfo("Grasped the block")
        sleep(0.5)

        # Lift up
        self.robotic_arm.move_single_servo(2, 70, 1500)
        rospy.loginfo("Lifted up")
        sleep(1)

        # Put down
        self.robotic_arm.move_single_servo(2, block_joints[1], 1500)
        rospy.loginfo("Moved to target location")
        sleep(1)

        # Release the block
        self.robotic_arm.move_single_servo(6, 30, 500)
        rospy.loginfo("Released the block")
        sleep(0.5)

        # Lift up
        self.robotic_arm.move_servos(joints_up, 1000)
        rospy.loginfo("Lifted up")
        sleep(1)
        
        # Reset
        self.robotic_arm.set_home_position()
