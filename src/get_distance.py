import pandas as pd
from scipy.spatial.distance import euclidean

# Load the data
file_path = 'data/profile-traffic-0.csv'  # Replace with the path to your file
traffic_data = pd.read_csv(file_path)

# Calculate the averages for each task (first metric)
task_averages = traffic_data.groupby('taskid').mean()[['incoming1', 'incoming2']]
import numpy as np


def vector_angle(v1, v2):
    """Calculate the cosine of the angle between two vectors."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(cos_angle, -1, 1))  # To handle numerical issues


# Initialize dictionaries to store the results for angle-based and distance-based metrics
closest_tasks_angle = {}
closest_tasks_distance = {}

# Iterate over each task for both angle and distance metrics
for current_task_id in task_averages.index:
    min_angle = float('inf')
    min_distance = float('inf')
    closest_task_angle = None
    closest_task_distance = None
    current_task_vector = task_averages.loc[current_task_id].values

    for previous_task_id in task_averages.index:
        if previous_task_id < current_task_id:
            previous_task_vector = task_averages.loc[previous_task_id].values

            # Calculate angle and Euclidean distance
            angle = vector_angle(current_task_vector, previous_task_vector)
            distance = euclidean(current_task_vector, previous_task_vector)

            # Update closest task based on angle
            if angle < min_angle:
                min_angle = angle
                closest_task_angle = previous_task_id

            # Update closest task based on distance
            if distance < min_distance:
                min_distance = distance
                closest_task_distance = previous_task_id

    closest_tasks_angle[current_task_id] = closest_task_angle
    closest_tasks_distance[current_task_id] = closest_task_distance

# Output the results
print("Closest Tasks Based on Angle:", closest_tasks_angle)
print("Closest Tasks Based on Eucli:", closest_tasks_distance)
