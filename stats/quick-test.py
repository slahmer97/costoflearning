import pandas as pd


data1 = pd.read_csv("7.39 9.5-7.39 9.5.csv", delimiter=" ")
data2 = pd.read_csv("7.39 9.5-7.39 10.0.csv", delimiter=" ")

print(data1.reward.mean())
print(data2.reward.mean())