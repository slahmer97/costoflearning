import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# Brighter color palette based on provided colors
custom_cycler = cycler(color=['#57a0ce', '#ff9f68', '#70d167', '#e77eb6', '#c2c2f0',
                              '#ae7268', '#f4b6d2', '#b0b0b0', '#dcd92a', '#4cd1c0'])
#plt.rcParams['text.usetex'] = True

plt.rc('axes', prop_cycle=custom_cycler)  # Set the custom color cycle
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2})

# Load the data (assuming the file and directory are correctly set)
traffic_data = pd.read_csv('data/profile-traffic-0.csv')

task_ids = [0, 1, 2, 3, 4]
filtered_data = traffic_data[traffic_data['taskid'].isin(task_ids)]
final_data = pd.DataFrame()

for task_id in task_ids:
    task_data = traffic_data[traffic_data['taskid'] == task_id]
    num_points = 2000
    group_size = len(task_data) // num_points
    max_values = task_data.groupby(np.arange(len(task_data)) // group_size).max()
    final_data = pd.concat([final_data, max_values.head(num_points)])

final_data.reset_index(drop=True, inplace=True)

final_data['incoming1_ma'] = final_data['incoming1']
final_data['incoming2_ma'] = final_data['incoming2']

plt.figure(figsize=(10,3))
plt.plot(final_data['incoming1_ma'], label='Active flows s_{0}')
plt.plot(final_data['incoming2_ma'], label='Active flows s_{1}')

legend_added_incoming1 = False
legend_added_incoming2 = False

for task_id in final_data['taskid'].unique():
    task_data = final_data[final_data['taskid'] == task_id]
    start_idx, end_idx = task_data.index[0], task_data.index[-1]
    task_avg_incoming1 = task_data['incoming1'].mean()
    task_avg_incoming2 = task_data['incoming2'].mean()

    plt.hlines(task_avg_incoming1, start_idx, end_idx, colors='#ff9f68', linestyles='dashed',
               label='Avg Incoming1 All Tasks' if not legend_added_incoming1 else "")
    legend_added_incoming1 = True

    plt.hlines(task_avg_incoming2, start_idx, end_idx, colors='#70d167', linestyles='dashed',
               label='Avg Incoming2 All Tasks' if not legend_added_incoming2 else "")
    legend_added_incoming2 = True

    if end_idx < len(final_data) - 1:
        plt.axvline(x=end_idx, color='grey', linestyle='--')

plt.ylabel('Active Flows')
plt.xticks([0, 2000, 4000, 8000, 10000], ['0', '500', '1000', '1500', '2000'], rotation=20)
plt.legend()
plt.grid(True)
plt.title('Instantaneous Number of Active Flows')
plt.tight_layout()

# Save as a TeX file
import tikzplotlib
tikzplotlib.save("tex-fig-output/active-flow.tex")

plt.show()