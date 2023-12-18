import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# Custom color palette
# Brighter color palette based on provided colors
custom_cycler = cycler(color=['#57a0ce', '#ff9f68', '#70d167', '#e77eb6', '#c2c2f0',
                              '#ae7268', '#f4b6d2', '#b0b0b0', '#dcd92a', '#4cd1c0'])
plt.rcParams['text.usetex'] = True

plt.rc('axes', prop_cycle=custom_cycler)  # Set the custom color cycle
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})
# Load the data
file_path = 'data/profile-traffic-0.csv'  # Replace with the path to your file
directory = 'data'

traffic_data = pd.read_csv(file_path)

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
plt.xticks([0, 500, 1000, 1500, 2000, 2500], ['0', '500', '1000', '1500', '2000', "2500"], rotation=20)

# Plot the EMA of the performance metric
plt.plot(combined_data['q1_ema'], label='\(|Q_{0}|\)', color='forestgreen')
plt.plot(combined_data['q2_ema'], label='\(|Q_{1}|\)', color='blue')

plt.ylabel('Normalized Queue Size')
plt.grid(True)
#plt.title('Queue size')

# Mark the boundaries of each task
for task_id in combined_data['task_id'].unique():
    end_idx = combined_data[combined_data['task_id'] == task_id].index[-1]
    plt.axvline(x=end_idx, color='red', linestyle='--', label=f'Task {task_id} End' if task_id == 0 else None)

plt.tight_layout()

# Save as a TeX file
import tikzplotlib

tikzplotlib.save("tex-fig-output/normalized-queue-size.tex")

plt.show()
