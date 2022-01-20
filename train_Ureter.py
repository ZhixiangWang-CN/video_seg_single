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




    train_ds,val_ds = read_data_ds(data_path=json_file,new_size=newsize)

    set_determinism(seed=0)

    val_loader = DataLoader(train_ds[:10], batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = DataLoader(train_ds[:30], batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device(device_type)

    model = load_model(name='UR')
    model.to(device)



    loss_function =  My_loss()
    optimizer = torch.optim.Adam(
        model.parameters(), learning_rate, weight_decay=1e-5, amsgrad=True
    )

    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    mean_dice_list = []

    time_save = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

    out_path = 'training_results/' + structure + '/' + time_save + '/'
    model_path = out_path + 'model/'
    pic_save_path = out_path + 'images/'
    model_load_path = 'training_results/' + structure+'/'+"best_metric_model.pth"

    if continue_training == True:


        model.load_state_dict(
            torch.load(model_load_path)
        )
        print(model)

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        train_bar = tqdm(train_loader)
        for batch_data in train_bar:

            step += 1

            inputs, labels = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                    )
            labels[labels==2]=0
            optimizer.zero_grad()
            outputs= model(inputs)
            # labels = labels.long()
            # outputs = torch.sigmoid(outputs)
            # outputs = torch.softmax(outputs,dim=1)
            # check_outputs(inputs,labels,outputs)
            # outputs=torch.squeeze(outputs,1)
            loss = loss_function(outputs,labels)

            loss.backward()
            optimizer.step()
            train_bar.set_description("loss: %s" % (loss.item()))
            epoch_loss += loss.item()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            dice_metric = DiceMetric(include_background=False, reduction="mean")
            model.eval()

            Dice_list = []

            print("evaluating.........")
            save_img_name = pic_save_path + str(epoch) + '_imgs.png'
            with torch.no_grad():

                val_bar = tqdm(val_loader)
                for val_data in val_bar:

                    val_inputs, val_labels = (
                        val_data['image'].to(device),
                        val_data["label"].to(device),
                    )
                    val_labels[val_labels==2]=0
                    val_outputs = model(val_inputs)
                    val_outputs = torch.softmax(val_outputs, dim=1)
                    # val_outputs = torch.where(val_outputs > 0.5, 1, 0)
                    # val_outputs=val_outputs.cpu().detach().numpy()
                    # val_labels=val_labels.cpu().detach().numpy()
                    # mask = onehot2mask(val_outputs[0])

                    # check_outputs(val_inputs,val_labels,val_outputs)#检测labels
                    # o_ = torch.softmax(val_outputs, dim=1)
                    val_outputs = torch.argmax(val_outputs, dim=1)
                    dice_score = dice_metric(val_outputs,val_labels)
                    val_outputs=val_outputs.cpu().detach().numpy()
                    val_labels=val_labels.cpu().detach().numpy()
                    # mask = onehot2mask(val_outputs[0])

                    # check_outputs(val_inputs,val_labels,val_outputs)#检测labels
                    # o_ = torch.softmax(val_outputs, dim=1)

                    Dice_list.append(dice_score[0].item())


                mean_dice = np.mean(Dice_list)
                if mean_dice > best_metric:
                    if not os.path.exists(pic_save_path):
                        os.makedirs(pic_save_path)
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    best_metric = mean_dice
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_path, "best_metric_model.pth"),
                    )

                    torch.save(
                        model.state_dict(),
                        model_load_path,
                    )
                    print("saved new best metric model")
                    # sum_list = sum(bn_iou_list)
                print(
                    f"current epoch: {epoch + 1} "
                )
                print("mean dice",mean_dice)
                mean_dice_list.append(mean_dice)

                val_save_name = pic_save_path + 'eval_loss.png'

                print("saving........", save_img_name)

                plot_evl_epochs(val_interval, epoch_loss_values, mean_dice_list, val_save_name)
                in_ = val_inputs[0].cpu().detach().numpy()
                # ou_ = val_outputs[0]
                lb_ = val_labels[0]
                # # mk = onehot2mask(ou_)
                mk = val_outputs[0]
                # mk_lb = onehot2mask(lb_)
                plot_mask_label_img_signal(in_, mk, lb_, save_name=save_img_name, dice_score=mean_dice)
                # plot_output(val_inputs[0], mask_,save_name=save_img_name,dice_list=IoU_List)

    print(
        f"train completed, best_metric: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters",type=str,default='3DUNet_UR.json', help="parameters json")
    args = parser.parse_args()


    para_path = './parameters/'+args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)