# This file needs to take in the joint_angles.json files
# Each data point needs to get a goal_joints fields
# where the goal_joints is calculated using interpolation between the current data point and next data points joint angles

import json
import os

output_file = "joint_angles_mod.json"

def interpolate_joints(current_joints, next_joints, last_valid_joints):
    """ Linearly interpolate between two sets of joint angles, handling None values, and convert to integers. """
    interpolated_joints = []
    for cj, nj, lvj in zip(current_joints, next_joints, last_valid_joints):
        if cj is None:
            cj = lvj
        if nj is None:
            nj = lvj
        if cj is None and nj is None:  # If both are None, use the last valid joint directly
            interpolated_joints.append(int(lvj))  # Convert to int
        else:
            interpolated_value = (cj + nj) / 2
            interpolated_joints.append(int(interpolated_value))  # Convert to int
    return interpolated_joints
    
def add_goal_joints_to_data(json_file):
    """ Add goal_joints to each data point in the JSON file, handling None values in joint angles. """
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    for iteration, records in data['data'].items():
        last_valid_joints = [0] * len(records[0]['joint_angles'])  # Initialize with zeros or another default value
        for i in range(len(records) - 1):
            current_joints = records[i]['joint_angles']
            next_joints = records[i + 1]['joint_angles']

            # Update last_valid_joints with current joint values if they are not None
            last_valid_joints = [cj if cj is not None else lvj for cj, lvj in zip(current_joints, last_valid_joints)]

            goal_joints = interpolate_joints(current_joints, next_joints, last_valid_joints)
            records[i]['goal_joints'] = goal_joints
        
        # Handle the last record: use its own joints as goal_joints, or last valid if None
        records[-1]['goal_joints'] = [j if j is not None else lvj for j, lvj in zip(records[-1]['joint_angles'], last_valid_joints)]

    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
json_file_path = '../data/bowl_data/diagonal/joint_angles.json'
add_goal_joints_to_data(json_file_path)
