import json

data_path = 'C:/Softwares/Codes/video_seg/Tools/patient_seprate_0106.json'
with open(data_path,'r') as load_f:
    load_dict = json.load(load_f)

train_list = [0,1,2,3,6]
test_list = [4,5]
train_l = []
val_l =[]
test_l = []
for i in train_list:
    train_l+=load_dict[str(i)+'_train']
    train_l+=load_dict[str(i)+'_test']
    val_l+=load_dict[str(i)+'_val']
test_1_l = []
test_2_l =[]


test_1_l += load_dict['4_train']
test_1_l += load_dict['4_test']
test_1_l += load_dict['4_val']

test_2_l += load_dict['5_train']
test_2_l += load_dict['5_test']
test_2_l += load_dict['5_val']

test_all = test_1_l+test_2_l
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

dicts = {
            "training": train_l, "test": test_all, "validation": val_l,'test1':test_1_l,'test2':test_2_l}

with open("dataset_only_UR_0119.json", "w") as dump_f:
    json.dump(dicts, dump_f, indent=4)