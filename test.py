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
import scipy

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


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

    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=0)


    device = torch.device(device_type)

    model = load_model()
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
    Dise_list_class1=[]
    Dise_list_class2 = []
    Dise_list_mean = []
    Haus_list_class1=[]
    Haus_list_class2=[]
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for val_data in val_bar:
            flag = 0
            val_inputs, val_labels = (
                val_data['image'].to(device),
                val_data["label"].to(device),
            )
            label_temp = val_data['label'].cpu().detach().numpy()
            label_temp2 = np.where(label_temp==2)
            n_2 = len(label_temp2[0])

            step+=1
            st= time.time()
            val_outputs = model(val_inputs)

            val_outputs = torch.softmax(val_outputs,dim=1)
            mask = torch.argmax(val_outputs, dim=1)
            mask_save_name = pic_save_path + str(step) + '_post.tif'
            # mask = post_processing(mask,tensor=True,save_name='')
            # mask = post_processing(mask, tensor=True)
            et = time.time()
            cost_time = et - st
            print("cost time", cost_time)
            if step >= 2:
                cost_time_list.append(cost_time)
            # val_outputs = torch.where(val_outputs > 0.5, 1, 0)
            dice_l,mean = cal_dice_seprate_mask(mask[0],val_labels[0])
            Haus_l =cal_Haus_seprate_mask(mask[0],val_labels[0])
            print("Dice",dice_l)
            print("Haus",Haus_l)
            if n_2 >100:
                d2 = dice_l[1]
                h2 = Haus_l[1]
                print("H2=====",h2)
                if d2>0.1:
                    Dise_list_class2.append(d2)
                # if h2>0 and h2<299:
                    Haus_list_class2.append(h2)

            d1 = dice_l[0]
            h1 = Haus_l[0]
            if d1>0.1:
                Dise_list_class1.append(d1)
            if h1>0 and h1 < 99:
                Haus_list_class1.append(h1)

            Dise_list_mean.append(mean)
            save_img_name = pic_save_path + str(step) + '_imgs.png'
            val_inputs = val_inputs.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            print("saving........", save_img_name)
            # save_test_result(val_inputs[0], mask, save_name=save_img_name, dice_list=dice_l)
        Dise_list_class1=np.array(Dise_list_class1)
        Dise_list_class2 = np.array(Dise_list_class2)
        Haus_list_class1 = np.array(Haus_list_class1)
        Haus_list_class2 = np.array(Haus_list_class2)
        cost_time_list = np.array(cost_time_list)
        print("AVG mean Dice class 1 ",Dise_list_class1.mean())
        print("AVG mean Dice class 2 ", Dise_list_class2.mean())
        print("AVG mean Dice ", (Dise_list_class1.mean()+Dise_list_class2.mean())/2)
        print("AVG mean Haus class 1 ", Haus_list_class1.mean())
        print("AVG mean Haus class 2 ", Haus_list_class2.mean())
        print("AVG mean Haus ", (Haus_list_class1.mean() + Haus_list_class2.mean()) / 2)
        print("Cost time each",cost_time_list.mean())
        Dise_list_class1 = [float(i) for i in Dise_list_class1]
        Dise_list_class2 = [float(x) for x in Dise_list_class2]
        cost_time_list = [float(x) for x in cost_time_list]
        Haus_list_class1 = [float(i) for i in Haus_list_class1]
        Haus_list_class2 = [float(x) for x in Haus_list_class2]

        D195 = mean_confidence_interval(Dise_list_class1)
        D295 = mean_confidence_interval(Dise_list_class2)
        H195 = mean_confidence_interval(Haus_list_class1)
        H295 = mean_confidence_interval(Haus_list_class2)
        print("D195",D195)
        print("D295",D295)
        print("H195",H195)
        print("H295",H295)
        test_metric = {'Dice1':list(Dise_list_class1),'Dice2':list(Dise_list_class2),'Haus1':list(Haus_list_class1),'Haus2':list(Haus_list_class2),'Time':list(cost_time_list)}
        json_str = json.dumps(test_metric, ensure_ascii=False, indent=4) # 缩进4字符





        with open(pic_save_path+'test_results_cpu_np_post.json', 'w') as json_file:
            json_file.write(json_str)
        plt.clf()
        plt.figure(figsize=(7, 5))
        n = len(Dise_list_class1)
        y = np.ones(n)
        y2 = np.ones(len(Dise_list_class2)) * 2
        plt.subplot(1,3,1)
        plt.hist(Dise_list_class1,color='red',alpha=0.5)
        plt.title("UR Histogram")
        plt.subplot(1,3,2)
        plt.scatter(y, Dise_list_class1, c='red', alpha=0.5)
        plt.scatter(y2, Dise_list_class2, c='blue', alpha=0.5)
        plt.boxplot([Dise_list_class1, Dise_list_class2], labels=['Ureter', 'UA'], showfliers=False)
        plt.title("Dice")
        plt.ylim((0, 1))
        plt.subplot(1,3,3)
        plt.hist(Dise_list_class2, color='blue', alpha=0.5)
        plt.title("UA Histogram")
        plt.savefig(pic_save_path+"Dice.tif")
        # plt.show()
        plt.clf()
        plt.figure(figsize=(7, 5))
        n = len(cost_time_list)
        y = np.ones(n)
        plt.subplot(1,2,1)
        plt.scatter(y, cost_time_list)
        plt.boxplot(cost_time_list, labels=['Time cost'], showfliers=False)
        plt.title("Time")
        plt.subplot(1, 2, 2)
        plt.hist(cost_time_list, alpha=0.5)
        plt.title("Histogram")
        # plt.ylim((0, 1))
        plt.savefig(pic_save_path+"Time.tif")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters",type=str,default='3DUNet_test.json', help="parameters json")
    args = parser.parse_args()


    para_path = './parameters/'+args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)