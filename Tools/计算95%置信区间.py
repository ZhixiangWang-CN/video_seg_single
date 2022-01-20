import numpy as np
import scipy.stats
import json

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

data_path = 'C:/Softwares/Codes/video_seg/Tools/UR_test_results.json'
with open(data_path,'r') as load_f:
    load_dict = json.load(load_f)
print(load_dict)
# Dice
data = load_dict['Dice']
# data1 = load_dict['Dice1']
# data2 = load_dict['Dice2']
# time_cost = load_dict['Time']
Haus = load_dict['Haus']
# Haus1 = load_dict['Haus1']
# Haus2 = load_dict['Haus2']
D95 = mean_confidence_interval(data)
# D195 = mean_confidence_interval(data1)
# D295 = mean_confidence_interval(data2)
# T95 = mean_confidence_interval(time_cost)
H95 = mean_confidence_interval(Haus)
# H195 = mean_confidence_interval(Haus1)
# H295 = mean_confidence_interval(Haus2)
print("D95",D95)

# print("D195",D195)
# print("D295",D295)
# print("T95",T95)
# print("H195",H195)
# print("H295",H295)
print("H95",H95)