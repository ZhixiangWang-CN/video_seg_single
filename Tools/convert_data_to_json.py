import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
def find_end(file_name):
    names = file_name.split("_")
    names_t = names[-1].split(".")
    return names_t[0]


def files_name(file_dir):
    data_list = []
    UR_list = []
    UA_list=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_name = os.path.join(root,file)
            file_name  = file_name.replace("\\","/")
            dict_temp = {}
            img_name = file_name
            label_name = file_name.replace('image','mask')
            label_name = label_name.replace('img', 'mask')
            label_name = label_name.replace('png', 'npy')
            dict_temp['image']=img_name
            dict_temp['label'] = label_name
            data_list.append(dict_temp)
            mask_data = np.load(label_name)
            UR_pos = np.where(mask_data==1)
            UA_pos = np.where(mask_data==2)
            if len(UR_pos[0])>200:
                UR_list.append(dict_temp)
            if len(UA_pos[0])>200:
                UA_list.append(dict_temp)

    return data_list,UR_list,UA_list



if __name__ == '__main__' :
    dicts =  {
        "name": "IMG",
        "description": "video",
        "reference": "",
        "licence":"",
        "release":"",
        "tensorImageSize": "4D",
        "modality": {
             "img": "image",
         },
         "labels": {
             "0": "background",
             "1": "Ureter",
             "2": "UA",
         } }



    file_dir_training = 'C:/Dataset/video/Segdataset/UR/image/'

    train_names,UR_names,UA_names = files_name(file_dir_training)
    random.shuffle(train_names)
    random.shuffle(UR_names)
    random.shuffle(UA_names)
    n = len(train_names)
    print("n=",n)
    train_set = train_names[:int(n*0.8)]
    test_set = train_names[int(n*0.8):]
    val_set = train_names[int(n*0.8):]
    # train_set = train_names[:int(n*0.8)]
    # test_set  = train_names[int(n*0.8):]
    # val_names = files_name(file_dir_val)
    dicts_train = {"nums:":n,"numTraining": int(n*0.8),"numval":n-int(n*0.8), "numTest":n-int(n*0.8), "training": train_set, "test": test_set, "validation": val_set}
    dicts.update(dicts_train)
    n=len(UR_names)
    dicts_UR = {"nums:": n, "numTraining": int(n * 0.8), "numval": n - int(n * 0.8), "numTest": n - int(n * 0.8),
                   "training": UR_names[:int(0.8*n)], "test":UR_names[int(0.8*n):], "validation": UR_names[int(0.8*n):]}
    n = len(UA_names)
    dicts_UA = {"nums:": n, "numTraining": int(n * 0.8), "numval": n - int(n * 0.8), "numTest": n - int(n * 0.8),
                "training": UA_names[:int(0.8 * n)], "test": UA_names[int(0.8 * n):],
                "validation": UA_names[int(0.8 * n):]}

    with open("dataset.json","w") as dump_f:
        json.dump(dicts,dump_f,indent=4)
    with open("dataset_UR.json", "w") as dump_f:
        json.dump(dicts_UR, dump_f, indent=4)
    with open("dataset_UA.json", "w") as dump_f:
        json.dump(dicts_UA, dump_f, indent=4)