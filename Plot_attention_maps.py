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
from gcam import gcam
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

    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


    device = torch.device(device_type)

    model_ = load_model()
    model_.to(device)

    time_save = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

    out_path = 'Test_results/' + structure + '/' + time_save + '/'

    pic_save_path = out_path + 'images/'
    #
    # if not os.path.exists(pic_save_path):
    #     os.makedirs(pic_save_path)

    model_.load_state_dict(
        torch.load(model_load_path + 'best_metric_model.pth')
    )
    print("=> loading checkpoint '{}'".format(model_load_path + 'best_metric_model.pth'))
    print(model_)
    checkpoint = torch.load(model_load_path + 'best_metric_model.pth')
    layers_list = []
    for k,v in checkpoint.items():
        print(k)
        if 'residual' in k:
            k=k.replace('.weight','')
            k = k.replace('.bias','')
            if k not in layers_list:
                layers_list.append(k)

        # print("Shape",v.shape)
    # layers_list=['model.0','model.1','model.2']


    output_dir = 'gcam/'
    for layer in layers_list:
        # model = gcam.inject(model_, output_dir='GBP/'+layer+'/', backend='gbp', layer=layer, label='best', save_maps=True)
        model = gcam.inject(model_, output_dir=output_dir + layer + '/', backend='gcam', layer=layer, label='best',
                            save_maps=True)
        # model = gcam.inject(model_, output_dir=output_dir + layer + '/', backend='ggcam', layer=layer, label='best',
        #                     save_maps=True)
        # model.eval()
        print("evaluating.........")

        with torch.no_grad():
            i=0
            val_bar = tqdm(val_loader)
            for val_data in val_bar:

                val_inputs, val_labels = (
                    val_data['image'].to(device),
                    val_data["label"].to(device),
                )
                image_dir = output_dir+'/image'
                image_name = image_dir + '/'+str(i)+'.jpg'
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                st= time.time()
                val_outputs = model(val_inputs)

                save_image(val_inputs[0],image_name)
                i+=1
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters",type=str,default='3DUNet_test.json', help="parameters json")
    args = parser.parse_args()


    para_path = './parameters/'+args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)