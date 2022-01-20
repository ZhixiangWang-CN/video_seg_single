import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from monai.apps import DecathlonDataset
from monai.config import print_config

from monai.data import DataLoader
from Utils.utils import *
from torch.nn import *
from monai.metrics import DiceMetric
import time
from monai.utils import set_determinism
import matplotlib.pyplot as plt
import torch
from model.load_model import load_model
from torch.autograd import Variable
from Loader.loader_img_test import read_data_ds
from monai.transforms import Compose,Activations,AsDiscrete
from Loss.load_loss import My_loss
import argparse
from tqdm import tqdm
import torch.nn.functional as F
def train(parameters):
    json_file =parameters["json_path"]

    newsize = parameters["Resize"]
    batch_size = parameters["batch_size"]
    structure =  parameters["structure"]
    device_type =  parameters["device_type"]
    model_load_path = parameters["model_load_path"]

    if model_load_path == 0:
        model_load_path='training_results/' + structure+'/'

    test_ds = read_data_ds(data_path=json_file,new_size=newsize)

    set_determinism(seed=0)

    val_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=0)


    device = torch.device(device_type)

    model = load_model(name=structure)
    model.to(device)

    time_save = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

    out_path = 'Test_results/' + structure + '/' + time_save + '/'

    pic_save_path = out_path + 'images/'

    if not os.path.exists(pic_save_path):
        os.makedirs(pic_save_path)

    model.load_state_dict(
        torch.load(model_load_path + '/best_metric_model.pth')
    )
    print(model)

    model.eval()
    print("evaluating.........")
    step = 0
    cost_time_list =[]
    Dise_list=[]
    Haus_list=[]
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for val_data in val_bar:
            flag = 0
            val_inputs, val_labels = (
                val_data['image'].to(device),
                val_data["label"].to(device),
            )
            val_labels[val_labels==2]=0

            step+=1
            st= time.time()
            val_outputs = model(val_inputs)
            # val_outputs=torch.sigmoid(val_outputs)
            # val_outputs = torch.where(val_outputs > 0.5, 1, 0)
            val_outputs = torch.softmax(val_outputs,dim=1)
            mask = torch.argmax(val_outputs, dim=1)
            # mask = torch.argmax(val_outputs, dim=1)
            # mask_save_name = pic_save_path + str(step) + '_post.tif'
            mask = post_processing(mask,tensor=True)
            # mask = post_processing(mask, tensor=True)
            et = time.time()
            cost_time = et - st
            print("cost time", cost_time)
            if step >= 2:
                cost_time_list.append(cost_time)

            dice_mean = dice_metric(mask,val_labels)
            Haus_score = cal_Haus_mask(mask[0],val_labels[0])
            dice_mean=dice_mean[0].item()
            if dice_mean>0.1:
                Dise_list.append(dice_mean)

                save_img_name = pic_save_path + str(step) + '_imgs.png'
                val_inputs = val_inputs.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()
            if Haus_score>0 and Haus_score<99:
                Haus_list.append(Haus_score)



                print("saving........", save_img_name)
                # save_test_result(val_inputs[0], mask, save_name=save_img_name, dice_list=dice_mean)
                # save_test_result_signal(val_inputs[0],mask,save_name=save_img_name,dice_score=dice_mean)
        Dise_list=np.array(Dise_list)
        Haus_list=np.array(Haus_list)
        cost_time_list = np.array(cost_time_list)
        print("AVG mean Dice ",Dise_list.mean())
        print("AVG mean Haus ",Haus_list.mean())
        print("Cost time each",cost_time_list.mean())
        Dise_list_class= [float(i) for i in Dise_list]
        Haus_list_class = [float(i) for i in Haus_list]
        cost_time_list = [float(x) for x in cost_time_list]
        test_metric = {'Dice':list(Dise_list_class),'Haus':Haus_list_class,'Time':list(cost_time_list)}
        json_str = json.dumps(test_metric, ensure_ascii=False, indent=4) # 缩进4字符
        with open(pic_save_path+'UR_test_results.json', 'w') as json_file:
            json_file.write(json_str)
        plt.clf()
        n = len(Dise_list)
        y = np.ones(n)
        plt.subplot(1,2,1)
        plt.scatter(y, Dise_list, c='red', alpha=0.5)

        plt.boxplot([Dise_list], labels=['UA'], showfliers=False)
        plt.title("UR Dice")
        plt.ylim((0, 1))
        plt.subplot(1,2,2)
        plt.hist(Dise_list,color='red' ,bins=20,edgecolor="black",alpha=0.7)
        plt.title("UR Dice Histogram")
        plt.savefig(pic_save_path + "Dice.tif")
        # plt.show()
        plt.clf()
        n = len(cost_time_list)
        y = np.ones(n)
        plt.subplot(1,2,1)
        plt.scatter(y, cost_time_list,c='red')
        plt.boxplot(cost_time_list, labels=['Time cost'], showfliers=False)
        plt.title("UR Time")
        # plt.ylim((0, 1))
        plt.subplot(1, 2, 2)
        plt.hist(cost_time_list, color='red', bins=20, edgecolor="black", alpha=0.5)
        plt.title("UR Time Histogram")
        # plt.savefig(pic_save_path + "Time.tif")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters",type=str,default='3DUNet_UR.json', help="parameters json")
    args = parser.parse_args()


    para_path = './parameters/'+args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)