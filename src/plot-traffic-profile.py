import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the data
file_path = 'data/profile-traffic-0.csv'  # Replace with the path to your file
traffic_data = pd.read_csv(file_path)

# Calculate the moving averages
window_size = 50  # Adjust as needed
traffic_data['incoming1_ma'] = traffic_data['incoming1'].rolling(window=window_size).mean()
traffic_data['incoming2_ma'] = traffic_data['incoming2'].rolling(window=window_size).mean()



# Plot the moving averages
plt.figure(figsize=(15, 7))
plt.plot(traffic_data['incoming1_ma'], label='Incoming1 MA')
plt.plot(traffic_data['incoming2_ma'], label='Incoming2 MA')

# Initialize flags for legend
legend_added_incoming1 = False
legend_added_incoming2 = False

# Calculate and plot the averages for each task, only over the task duration
for task_id in traffic_data['taskid'].unique():
    task_data = traffic_data[traffic_data['taskid'] == task_id]
    start_idx, end_idx = task_data.index[0], task_data.index[-1]
    task_avg_incoming1 = task_data['incoming1'].mean()
    task_avg_incoming2 = task_data['incoming2'].mean()

    # Plot avg lines with legend labels added only once
    if not legend_added_incoming1:
        plt.hlines(task_avg_incoming1, start_idx, end_idx, colors='purple', linestyles='dashed',
                   label='Avg Incoming1 All Tasks')
        legend_added_incoming1 = True
    else:
        plt.hlines(task_avg_incoming1, start_idx, end_idx, colors='purple', linestyles='dashed')

    if not legend_added_incoming2:
        plt.hlines(task_avg_incoming2, start_idx, end_idx, colors='cyan', linestyles='dashed',
                   label='Avg Incoming2 All Tasks')
        legend_added_incoming2 = True
    else:
        plt.hlines(task_avg_incoming2, start_idx, end_idx, colors='cyan', linestyles='dashed')

    # Mark task ID changes
    if end_idx < len(traffic_data) - 1:
        plt.axvline(x=end_idx, color='red', linestyle='--')

plt.title('Moving Averages of Incoming Packets with Task Averages')
plt.xlabel('Unit of Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
