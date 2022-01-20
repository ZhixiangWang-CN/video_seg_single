import json
import numpy as np
import matplotlib.pyplot as plt

path= 'C:/Softwares/Codes/video_seg/Tools/dataset_UR.json'
with open(path,'r') as load_f:
    load_dict = json.load(load_f)

cal_label=1 #1 = UR, 2=UA
nums_list =[]
sets = ['training','test','validation']
for name in sets:
    datas = load_dict[name]
    for data in datas:
        labels_p = data['label']
        label = np.load(labels_p)
        nums_l = np.where(label==cal_label)
        nums = len(nums_l[0])
        nums_list.append(nums)

nums_list=np.array(nums_list)

max_nums_list = nums_list.max()
min_nums_list = nums_list.min()
avg = np.mean(nums_list)
nums_list = list(nums_list)
print('max,min,avg',max_nums_list,min_nums_list,avg)

