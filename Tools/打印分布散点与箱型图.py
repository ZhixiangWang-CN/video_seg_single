import json
import matplotlib.pyplot as plt
import numpy as np

data_path = 'C:/Softwares/Codes/video_seg/Test_results/3DUNet/2021_09_05_18_35/images/test_results.json'
with open(data_path,'r') as load_f:
    load_dict = json.load(load_f)
print(load_dict)
# Dice
data1 = load_dict['Dice1']
data2 = load_dict['Dice2']
time_cost = load_dict['Time']
print("dice1 avg",np.array(data1).mean())
print("dice2 avg",np.array(data2).mean())
print("time avg",np.array(time_cost).mean())
print("mean",(np.array(data2).mean()+np.array(data2).mean())/2)
n = len(data1)
y = np.ones(n)
plt.figure(figsize=(7,5))
plt.subplot(1,3,1)

plt.hist(data1,color='red',edgecolor="black",alpha=0.5)
# plt.xlim((0.3, 1))
my_x_ticks = np.arange(0.3, 1, 0.2)
plt.xticks(my_x_ticks)
plt.title("UR Dice Histogram")
plt.subplot(1,3,2)
y2 = np.ones(len(data2))*2
plt.scatter(y,data1,c='red',alpha=0.5)
plt.scatter(y2,data2,c='blue',alpha=0.5)
plt.boxplot([data1,data2],labels=['Ureter','UA'],showfliers=False)
plt.title("Dice")
plt.ylim((0, 1))
plt.subplot(1,3,3)

plt.hist(data2,color='blue',edgecolor="black",alpha=0.5)
# plt.xlim((0.3, 1))
my_x_ticks = np.arange(0.3, 1, 0.2)
plt.xticks(my_x_ticks)
plt.title("UA Dice Histogram")
plt.savefig("Dice.tif")
# plt.show()
plt.clf()
# Time cost
plt.figure(figsize=(7, 5))
n = len(time_cost)
y = np.ones(n)
plt.subplot(1,2,1)
plt.scatter(y,time_cost)
plt.boxplot(time_cost,labels=['Time cost'],showfliers=False)
plt.title("Time")
# plt.ylim((0, 1))
plt.subplot(1,2,2)
plt.hist(time_cost, bins=20,edgecolor="black",alpha=0.7)
plt.title("Histogram")
plt.savefig("Time.tif")

