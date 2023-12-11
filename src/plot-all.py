import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# Custom color palette
custom_cycler = cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

plt.rc('axes', prop_cycle=custom_cycler)  # Set the custom color cycle
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})  # Adjust font size and line width

# Load the data
file_path = 'data/profile-traffic-0.csv'  # Replace with the path to your file
directory = 'data'

traffic_data = pd.read_csv(file_path)

task_ids = [0, 1, 2, 3, 4]
filtered_data = traffic_data[traffic_data['taskid'].isin(task_ids)]
final_data = pd.DataFrame()
for task_id in task_ids:
    task_data = traffic_data[traffic_data['taskid'] == task_id]

    # Number of data points to select from each task
    num_points = 10000

    # Divide the task data into 500 groups and select the maximum value from each group
    group_size = len(task_data) // num_points
    max_values = task_data.groupby(np.arange(len(task_data)) // group_size).max()

    # Append the downsampled data to final_data
    final_data = pd.concat([final_data, max_values.head(num_points)])

# Reset the index for plotting
final_data.reset_index(drop=True, inplace=True)

window_size = 2  # Adjust as needed
final_data['incoming1_ma'] = final_data['incoming1']#.rolling(window=window_size).mean()
final_data['incoming2_ma'] = final_data['incoming2']#.rolling(window=window_size).mean()

print(len(final_data["incoming1_ma"]))

fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(12, 6))

axs[0].plot(final_data['incoming1_ma'], label='Active flows s_{0}', color='royalblue')
axs[0].plot(final_data['incoming2_ma'], label='Active flows s_{1}', color='darkorange')
axs[0].set_ylabel('Active Flows')
axs[0].legend()
axs[0].grid(True)
axs[0].set_title('Instanteneous number of active flows ')

legend_added_incoming1 = False
legend_added_incoming2 = False
""""""
traffic_data = final_data
# Calculate and plot the averages for each task, only over the task duration
for task_id in traffic_data['taskid'].unique():
    task_data = traffic_data[traffic_data['taskid'] == task_id]
    start_idx, end_idx = task_data.index[0], task_data.index[-1]
    task_avg_incoming1 = task_data['incoming1'].mean()
    task_avg_incoming2 = task_data['incoming2'].mean()

    # Plot avg lines with legend labels added only once
    if not legend_added_incoming1:
        axs[0].hlines(task_avg_incoming1, start_idx, end_idx, colors='purple', linestyles='dashed',
                      label='Avg Incoming1 All Tasks')
        legend_added_incoming1 = True
    else:
        axs[0].hlines(task_avg_incoming1, start_idx, end_idx, colors='purple', linestyles='dashed')

    if not legend_added_incoming2:
        axs[0].hlines(task_avg_incoming2, start_idx, end_idx, colors='cyan', linestyles='dashed',
                      label='Avg Incoming2 All Tasks')
        legend_added_incoming2 = True
    else:
        axs[0].hlines(task_avg_incoming2, start_idx, end_idx, colors='cyan', linestyles='dashed')

    # Mark task ID changes
    if end_idx < len(traffic_data) - 1:
        axs[0].axvline(x=end_idx, color='red', linestyle='--')

combined_data = pd.DataFrame()

# Iterate through each file in the directory and combine data
dirs = sorted(os.listdir(directory))
for filename in dirs:
    if filename.startswith('task') and filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        task_data = pd.read_csv(file_path)
        task_data['performance'] = (task_data['reward'] + 250) / 250
        # Extract the task ID from the filename and add it as a column
        task_id = int(re.search(r'task-(\d+)-0\.csv', filename).group(1))
        print(task_id, task_data["performance"].mean())
        print(f"task-id={task_id}\n"
              f"\tperf={task_data['performance'].mean()}\n"
              f"\trejection={task_data['rejection1'].mean()}\n"
              f"\tdrop={task_data['drop2'].mean()}\n"
              f"\tlatency2={task_data['latency2'].mean()}\n"
              )
        task_data['task_id'] = task_id

        # Concatenate the task data
        combined_data = pd.concat([combined_data, task_data], ignore_index=True)

# Calculate the performance metric for the combined data
# combined_data['performance'] = (combined_data['reward'] + 250) / 250
combined_data['q1'] = combined_data['q1'] / 1500.0
combined_data['q2'] = combined_data['q2'] / 350.0
combined_data['tde'] = combined_data['tde'] / 1000.0

# Calculate the Exponential Moving Average (EMA) of the performance
ema_span = 40  # Define the span for EMA calculation, adjust as needed
combined_data['performance_ema'] = combined_data['performance'].ewm(span=30, adjust=False).mean()
combined_data['q1_ema'] = combined_data['q1'].ewm(span=5, adjust=False).mean()
combined_data['q2_ema'] = combined_data['q2'].ewm(span=5, adjust=False).mean()
combined_data['tde_ema'] = combined_data['tde'].rolling(window=10).mean()

# Plot the EMA of the performance metric
axs[1].plot(combined_data['performance_ema'], label='Reward', color='forestgreen')
axs[1].set_ylabel('Performance')
axs[1].legend()
axs[1].grid(True)
axs[1].set_title('EMA of Performance Over Time Across All Tasks')


axs[2].plot(combined_data['q1_ema'], label='Q1 EMA', color='midnightblue')
axs[2].plot(combined_data['q2_ema'], label='Q2 EMA', color='crimson')
axs[2].set_ylabel('Queue size (pkts)')
axs[2].legend()
axs[2].grid(True)
axs[2].set_title('EMA of Queue Lengths Over Time')


axs[3].plot(combined_data['tde_ema'], label='TDE EMA', color='purple')
axs[3].set_xlabel('Time (seconds)')
axs[3].set_ylabel('TDE')
axs[3].legend()
axs[3].grid(True)
axs[3].set_title('Temporal Difference Error (TDE) Over Time')

# Mark the boundaries of each task
for task_id in combined_data['task_id'].unique():
    end_idx = combined_data[combined_data['task_id'] == task_id].index[-1]
    axs[1].axvline(x=end_idx, color='red', linestyle='--', label=f'Task {task_id} End' if task_id == 0 else None)

plt.tight_layout()
plt.show()