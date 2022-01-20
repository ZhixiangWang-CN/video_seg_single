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
    data_list=[]
    file_list = os.listdir(file_dir)
    file_list.sort(key=lambda x: int(x[:-4]))

    for file in file_list:
        file_name = os.path.join(file_dir,file)
        file_name  = file_name.replace("\\","/")
        dict_temp = {}
        img_name = file_name

        dict_temp['image']=img_name

        data_list.append(dict_temp)


    return data_list



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



    file_dir_training = 'C:/Dataset/video/test/Test_img/'

    train_names= files_name(file_dir_training)

    n = len(train_names)
    print("n=",n)
    train_set = train_names


    dicts_train = {"nums:":n,"numtest": int(n), "test": train_set,"training": train_set,"validation": train_set}
    dicts.update(dicts_train)


    with open("dataset_test.json","w") as dump_f:
        json.dump(dicts_train,dump_f,indent=4)
