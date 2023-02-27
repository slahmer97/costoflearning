import numpy as np

my_cap = []

full_capacity = 1.0
for i in range(30):
    my_cap.append(1.0 - np.random.uniform(0.0, 0.13))

print(my_cap)

