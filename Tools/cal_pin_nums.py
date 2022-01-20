import json

data_path = 'C:/Softwares/Codes/video_seg/Tools/patient_seprate_0106.json'
with open(data_path,'r') as load_f:
    load_dict = json.load(load_f)

img_list =load_dict['training']
img_list+=load_dict['validation']
img_list+=load_dict['test']

n = len(img_list)
print("N=",n)