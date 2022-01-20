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
from Loader.loader_img_test_nolabel import read_data_ds
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

    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


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

    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for val_data in val_bar:
            flag = 0
            val_inputs, val_labels = (
                val_data['image'].to(device),
                val_data["image"].to(device),
            )


            step+=1
            st= time.time()
            val_outputs = model(val_inputs)

            val_outputs = torch.softmax(val_outputs,dim=1)
            mask = torch.argmax(val_outputs, dim=1)
            mask = post_processing(mask,tensor=True,save_name='')
            # mask = post_processing(mask, tensor=True)
            et = time.time()
            cost_time = et - st
            print("cost time", cost_time)

            save_img_name = pic_save_path + str(step) + '.jpg'
            val_inputs = val_inputs.cpu().detach().numpy()
            mask = mask.cpu().detach().numpy()
            print("saving........", save_img_name)
            save_test_result(val_inputs[0], mask, save_name=save_img_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters",type=str,default='3DUNet_test_nolabel.json', help="parameters json")
    args = parser.parse_args()


    para_path = './parameters/'+args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)