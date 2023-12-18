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
file_path = 'data/rho_evo-task-0.csv'  # Replace with the path to your file
directory = 'data'

traffic_data = pd.read_csv(file_path)

plt.figure(figsize=(16,4))

plt.plot(traffic_data['rho'], label='\(|Q_{0}|\)', color='forestgreen')

plt.ylabel('Normalized Queue Size')
plt.grid(True)
#plt.title('Queue size')



# Save as a TeX file
import tikzplotlib

#tikzplotlib.save("tex-fig-output/normalized-queue-size.tex")

plt.show()
