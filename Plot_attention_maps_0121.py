import os
import shutil
import tempfile
from pytorch_grad_cam import GradCAM
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
from Loader.loader_img import read_data_ds
from monai.transforms import Compose,Activations,AsDiscrete
from Loss.load_loss import My_loss
import argparse
from tqdm import tqdm
from gcam import gcam
import torch.nn.functional as F
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image



def show_cam_on_image_local(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    maxx = np.max(img)
    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]



class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        result = (model_output[self.category, :, :] * self.mask)

        result = result.sum()
        return result
def train(parameters):
    json_file =parameters["json_path"]

    newsize = parameters["Resize"]
    batch_size = parameters["batch_size"]
    structure =  parameters["structure"]
    device_type =  parameters["device_type"]
    model_load_path = parameters["model_load_path"]

    if model_load_path == 0:
        model_load_path='training_results/' + structure+'/'

    # test_ds = read_data_ds(data_path=json_file,new_size=newsize)
    #
    # set_determinism(seed=0)
    #
    # val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    train_ds,val_ds = read_data_ds(data_path=json_file,new_size=newsize)

    set_determinism(seed=0)

    val_loader = DataLoader(val_ds[:3], batch_size=1, shuffle=True, num_workers=0)
    train_loader = DataLoader(train_ds[:3], batch_size=1, shuffle=True, num_workers=0)

    device = torch.device(device_type)

    model = load_model(name='UR')
    model.to(device)

    time_save = time.strftime("%Y_%m_%d_%H_%M", time.localtime())

    out_path = 'Test_results/' + structure + '/' + time_save + '/'

    pic_save_path = out_path + 'images/'
    #
    # if not os.path.exists(pic_save_path):
    #     os.makedirs(pic_save_path)

    model.load_state_dict(
        torch.load(model_load_path + 'best_metric_model.pth')
    )
    print("=> loading checkpoint '{}'".format(model_load_path + 'best_metric_model.pth'))
    print(model)
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


    output_dir = './Feature_map/gcam/'





    for layer in layers_list:
        target_layers = [model.model[-1]]
        print("evaluating.........")


        i=0
        val_bar = tqdm(train_loader)
        for val_data in val_bar:

            val_inputs, val_labels = (
                val_data['image'].to(device),
                val_data["label"].to(device),
            )
            val_labels[val_labels==2]=0
            image_dir = output_dir+'/image'
            image_name = image_dir + '/'+str(i)+'.jpg'
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            val_labels=val_labels[0].cpu().detach().numpy()
            targets = [SemanticSegmentationTarget(category=0, mask=val_labels)]
            with GradCAM(model=model,
                         target_layers=target_layers,
                         use_cuda=torch.cuda.is_available()) as cam:
                grayscale_cam = cam(input_tensor=val_inputs,
                                    targets=targets)
                # val_inputs[val_inputs>1]=1
                val_inputs= val_inputs.cpu().detach().numpy()

                # cam_image = show_cam_on_image_local(val_inputs, grayscale_cam[0], use_rgb=True)
                plt.imshow(grayscale_cam[0])
                plt.show()
            # save_image(val_inputs[0],image_name)
            i+=1
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--parameters",type=str,default='3DUNet_UR.json', help="parameters json")
    args = parser.parse_args()


    para_path = './parameters/'+args.parameters

    with open(para_path, 'r') as f:  # 读取json文件并返回data字典
        data_para = json.load(f)
    train(data_para)