import cv2 as cv
import threading
from time import sleep
import rospy
from pick_and_place import PickAndPlace
from data_collection_new import DataCollection

exit_flag = False

def game_loop(game, color_hsv, data_collection):
    global exit_flag
    rospy.on_shutdown(shutdown_hook)

    # Initialize camera on robotic arm
    capture = cv.VideoCapture(0)
    task_iteration = 0
    
    while capture.isOpened() and not exit_flag:
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
    
    # Red block
    color_hsv = [(0, 100, 100), (10, 255, 255)]

    # Initialize DataCollection
    data_collection = DataCollection(1, "images", "control_signals.csv")
    data_collection.calibrate_camera()
    data_capture_frequency = 10  # Capture data 10 times per iteration
    data_collection.start_data_collection(game, data_capture_frequency)
    
    game_thread = threading.Thread(target=game_loop, args=(game, color_hsv, data_collection))
    game_thread.start()
    
    try:
        while game_thread.is_alive():
            game_thread.join()
    except KeyboardInterrupt:
        global exit_flag
        exit_flag = True
        print("Interrupted by user")
    finally:
        data_collection.stop_data_collection()

if __name__ == '__main__':
    main()
