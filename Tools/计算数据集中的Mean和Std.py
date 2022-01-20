import json
from PIL import Image
import numpy as np
data_path = 'dataset.json'
with open(data_path) as jsonFile:
    jsonList= json.load(jsonFile)

train_list = jsonList['training']
val_list = jsonList['validation']
all_files = train_list+val_list
means = [0, 0, 0]
stdevs = [0, 0, 0]
nums = len(all_files)
for i in range(nums):
    print(str(i)+'/'+str(nums))
    img_path = all_files[i]['image']
    im = Image.open(img_path)
    im_array = np.array(im)
    for j in range(3):
        k = im_array[:,:,j]/255#RGB
        mean = k.mean()
        print("mean",mean)

        std = k.std()
        print("std", std)
        means[j]+=mean
        stdevs[j]+=std
means = np.asarray(means) / nums
stdevs = np.asarray(stdevs) / nums

print("Mean = {}".format(means))
print("Std = {}".format(stdevs))

save_dict={"Dataset":"Urter&UA","Range":"RGB",'Means':list(means),"Stds":list(stdevs)}

with open("Mean_std.json","w") as f:
    json.dump(save_dict,f)
    print("加载入文件完成...")
