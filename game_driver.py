import cv2 as cv
import threading
from time import sleep
import rospy
from pick_and_place import PickAndPlace
from data_collection import DataCollection

exit_flag = False

def game_loop(game, color_hsv, data_collection):
    global exit_flag
    rospy.on_shutdown(shutdown_hook)

    # Initialize camera on robotic arm
    capture = cv.VideoCapture(0)
    task_iteration = 0
    
    while capture.isOpened() and not exit_flag and not rospy.is_shutdown():
        _, img = capture.read()
        img = cv.resize(img, (640, 480))

        # Apply perspective transformation
        if game.dp is not None:
            img = game.calibration.Perspective_transform(game.dp, img)
        cv.imshow("Camera Feed", img)
        
        block_position = game.get_block_position(img, color_hsv)
        if block_position is not None:
            # Calculate inverse kinematics for the block position
            joints = game.calculate_inverse_kinematics(block_position)
            
            if joints is not None:
                data_collection.start_task_data_collection()
                game.move_block(joints)
                data_collection.stop_task_data_collection(task_iteration)
                task_iteration += 1

                # Reset
                joints = None
                block_position = None
                sleep(3)
        else:
            rospy.loginfo("No block detected")
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True
        
        rospy.loginfo("Iteration completed")
        sleep(0.1)

    # Release the camera
    capture.release()

def shutdown_hook():
    global exit_flag
    exit_flag = True
    print("ROS node is shutting down")

def main():
    game = PickAndPlace()
    
    # Green block
    color_hsv = [(35, 43, 46), (77, 255, 255)]

    # Initialize DataCollection
    data_collection = DataCollection(camera_index=1)
    data_collection.calibrate_camera()
    data_capture_frequency = 10
    
    game_thread = threading.Thread(target=game_loop, args=(game, color_hsv, data_collection))
    game_thread.start()
    
    try:
        game_thread.join()
    except KeyboardInterrupt:
        global exit_flag
        exit_flag = True
        print("Interrupted by user")
        game_thread.join()
    finally:
        data_collection.stop_task_data_collection(-1)
        print("Data collection stopped in finally")

if __name__ == '__main__':
    main()
