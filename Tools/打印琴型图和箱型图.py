import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
data_path = './test_results_cpu_np_withpost0119.json'
with open(data_path,'r') as load_f:
    load_dict = json.load(load_f)
print(load_dict)
# Dice

data1 = load_dict['Dice1']
data2 = load_dict['Dice2']
time_cost = load_dict['Time']

data1 = np.array(data1)
data1 = data1[data1>0.4]

print("dice1 avg",np.array(data1).mean())
print("dice2 avg",np.array(data2).mean())
print("time avg",np.array(time_cost).mean())
print("mean",(np.array(data2).mean()+np.array(data2).mean())/2)
n = len(data1)
y = np.ones(n)
print(max(data1))
data = {'DICE score':data1,
       'Class':'UR',
       }
df1 = pd.DataFrame(data)
data = {'DICE score':data2,
       'Class':'UA',
       }
df2 = pd.DataFrame(data)

df = pd.concat([df1,df2],axis=0)

# plt.figure(figsize=(7,5))
# plt.subplot(1,2,1)
# y2 = np.ones(len(data2))*2
sns.violinplot(x="Class",y="DICE score",inner="box",width=0.4,cut=1,data=df)
# plt.violinplot([data1,data2],c=['r','b'],showmeans=True,showmedians=True)
plt.ylim((0, 1.1))
plt.title("DICE coefficient ")
# plt.savefig("violinplot_Dice1103.tif")
plt.show()
plt.clf()
# Time cost
# plt.figure(figsize=(7, 5))
# n = len(time_cost)
# y = np.ones(n)
# plt.subplot(1,2,1)
# plt.scatter(y,time_cost)
# plt.boxplot(time_cost,labels=['Time cost'],showfliers=False)
# plt.title("Time")
# # plt.ylim((0, 1))
# plt.subplot(1,2,2)
# plt.hist(time_cost, bins=20,edgecolor="black",alpha=0.7)
# plt.title("Histogram")
data_path = './test_results_cpu.json'
with open(data_path,'r') as load_f:
    load_dict = json.load(load_f)
    print(load_dict)
time_cost_cpu = load_dict['Time']


data = {'Seconds':time_cost,
        'Class':'GPU',
       }
df1 = pd.DataFrame(data)
data = {'Seconds':time_cost_cpu,
       'Class':'CPU',
       }
df2 = pd.DataFrame(data)

df = pd.concat([df1,df2],axis=0)
# df = pd.DataFrame(data)
sns.violinplot(x="Class",y="Seconds",inner="box",width=0.4,data=df)
# plt.show()
plt.title("Time consumption")
plt.savefig("violinplot_Time_cpu_GPU.tif")

