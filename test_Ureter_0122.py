import os
import shutil
import tempfile
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.losses import *
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
from Loader.loader_img import read_data_ds
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
    learning_rate = parameters["learning_rate"]
    max_epochs = parameters["max_epochs"]
    continue_training = parameters["continue_training"]
    device_type =  parameters["device_type"]




    train_ds,val_ds = read_data_ds(data_path=json_file,section='test',new_size=newsize)

    set_determinism(seed=0)

    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=0)


    device = torch.device(device_type)

    model = load_model(name='UR')
    model.to(device)





    time_save = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

    out_path = 'test_results/' + structure + '/' + time_save + '/'
    model_path = out_path + 'model/'
    pic_save_path = out_path + 'images/'
    if not os.path.exists(pic_save_path):
        os.makedirs(pic_save_path)
    model_load_path = 'training_results/' + structure+'/'+"best_metric_model.pth"




    model.load_state_dict(
        torch.load(model_load_path)
    )
    print(model)




    dice_metric = DiceMetric(include_background=False, reduction="mean")
    model.eval()

    Dice_list = []

    print("evaluating.........")

    with torch.no_grad():

        val_bar = tqdm(val_loader)
        i=0
        for val_data in val_bar:
            try:
                val_inputs, val_labels = (
                    val_data['image'].to(device),
                    val_data["label"].to(device),
                )
                val_labels[val_labels==2]=0
                val_outputs = model(val_inputs)
                # val_outputs = torch.softmax(val_outputs, dim=1)
                # val_outputs = torch.argmax(val_outputs, dim=1)
                val_outputs=torch.sigmoid(val_outputs)
                val_outputs = torch.where(val_outputs > 0.5, 1, 0)
                val_outputs=val_outputs[:,0,:,:]
                # val_outputs=val_outputs.cpu().detach().numpy()
                # val_labels=val_labels.cpu().detach().numpy()
                # mask = onehot2mask(val_outputs[0])

                # check_outputs(val_inputs,val_labels,val_outputs)#检测labels
                # o_ = torch.softmax(val_outputs, dim=1)
                # val_outputs = torch.argmax(val_outputs, dim=1)
                # val_outputs = torch.squeeze(val_outputs,dim=1)

                dice_score = dice_metric(val_outputs,val_labels)
                val_outputs=val_outputs.cpu().detach().numpy()
                val_labels=val_labels.cpu().detach().numpy()
                dice_score=dice_score[0].item()
                print("Dice score",dice_score)
                # mask = onehot2mask(val_outputs[0])

                # check_outputs(val_inputs,val_labels,val_outputs)#检测labels
                    # o_ = torch.softmax(val_outputs, dim=1)

                Dice_list.append(dice_score)
                save_image_name = pic_save_path+str(i)+'.png'
                print("saving........", save_image_name)
                in_ = val_inputs[0].cpu().detach().numpy()
                # ou_ = val_outputs[0]
                lb_ = val_labels[0]
                # # mk = onehot2mask(ou_)
                mk = val_outputs[0]
                # mk_lb = onehot2mask(lb_)
                plot_mask_label_img_signal(in_, mk, lb_, save_name=save_image_name, dice_score=dice_score)
                # plot_output(val_inputs[0], mask_,save_name=save_img_name,dice_list=IoU_List)
                i+=1
            except:
                pass
    mean_dice = np.mean(Dice_list)
    print("Mean Dice",mean_dice)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters",type=str,default='3DUNet_UR.json', help="parameters json")
    args = parser.parse_args()


    para_path = './parameters/'+args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)