import pydicom
import numpy as np
import os
from PIL import Image
import torch
from monai.data import CacheDataset
import matplotlib.pyplot as plt
import sys
from monai.transforms import *
from skimage import transform
import math
import skimage.morphology as sm
import numpy as np
from monai.config import *
import cv2
from skimage import morphology
from Tools import surface_distance
import pandas as pd
from sklearn.metrics import *
from typing import *
import random
from Utils.BraTS_datalist import *
from skimage import measure
import os
import re

def find_all_files(path):
    files_list =[]
    for root,dirs,files in os.walk(path):
        for file in files:
            name = os.path.join(root, file)
            name= name.replace('\\','/')
            files_list.append(name)
    return files_list




def mask2onehot(mask, num_classes=3):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    # try:
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    # except:
    #     _mask= torch.argmax(mask, axis=0)
        # _mask = torch.tensor(_mask,dtype=torch.uint8)
    return _mask

def reverse_norm(image):
    means = [0.7335126656833895, 0.5022088988016531, 0.46472987089654316]
    stds = [0.1836286815500267,0.2164973848121452,0.20074602079697546]
    for i in range(3):
        img_t = image[i]
        mean = means[i]
        std = stds[i]
        img_ = (img_t*std+mean)*255

        image[i] = img_
    return image
def save_test_result(image,mask,save_name,dice_list=''):
    # image = image*255

    image = reverse_norm(image)


    image = np.moveaxis(image, 0, -1)
    mask = np.expand_dims(mask[0], 0).repeat(3, axis=0)
    mask = np.moveaxis(mask, 0, -1)
    new_shape = [1080,1920,3]

    New_image =  transform.resize(image, new_shape, anti_aliasing=True, preserve_range=True)
    New_mask  =  transform.resize(mask, new_shape,order=0,preserve_range=True)
    New_mask1 = New_mask.copy()
    New_mask2 = New_mask.copy()
    New_mask1[:,:,0]=0
    New_mask1[:, :, 2] = 0
    New_mask2[:, :, 0] = 0
    New_mask2[:, :, 1] = 0
    New_image[New_mask1==1]=255
    New_image[New_mask2==2]=255
    New_image = np.flip(New_image, 2)
    New_image = cv2.UMat(New_image)
    if len(dice_list)>0:
        if dice_list[0]!=0 and dice_list[1]!=0:
            words = 'Dice UR,UA: ' + str(np.round(dice_list[0], 2)) + ',' + str(np.round(dice_list[1], 2))
        elif dice_list[0]!=0:
            words = 'Dice UR: ' + str(np.round(dice_list[0], 2))
        elif dice_list[1]!=0:
            words = 'Dice UA: ' + str(np.round(dice_list[1], 2))
        else:
            words=''
        cv2.putText(New_image, words, (150, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
    # New_image = np.flip(New_image, 2)
    cv2.imwrite(save_name, New_image)

def save_image(image,save_name):
    # image = image*255
    image = image.cpu().detach().numpy()
    image = reverse_norm(image)


    image = np.moveaxis(image, 0, -1)

    new_shape = [1080,1920,3]

    New_image =  transform.resize(image, new_shape, anti_aliasing=True, preserve_range=True)

    New_image = np.flip(New_image, 2)
    New_image = cv2.UMat(New_image)
    cv2.imwrite(save_name, New_image)


def save_test_result_signal(image,mask,dice_score,save_name):
    image = reverse_norm(image)
    image = np.moveaxis(image, 0, -1)
    mask = np.expand_dims(mask[0], 0).repeat(3, axis=0)
    mask = np.moveaxis(mask, 0, -1)
    new_shape = [1080,1920,3]

    New_image =  transform.resize(image, new_shape, anti_aliasing=True, preserve_range=True)
    New_mask  =  transform.resize(mask, new_shape,order=0,preserve_range=True)
    New_mask1 = New_mask.copy()
    New_mask2 = New_mask.copy()
    New_mask1[:,:,0]=0
    New_mask1[:, :, 2] = 0
    New_mask2[:, :, 0] = 0
    New_mask2[:, :, 1] = 0
    New_image[New_mask1==1]=255
    New_image[New_mask2==2]=255
    New_image = np.flip(New_image, 2)
    New_image = cv2.UMat(New_image)
    if dice_score!= 0:
        words  = 'DICE : ' + str(dice_score)

    else:
        words=''
    cv2.putText(New_image, words, (150, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 3)
    # New_image = np.flip(New_image, 2)
    cv2.imwrite(save_name, New_image)

def cal_dice2(pred,target):

    ep=1e-8
    intersection = 2 * torch.sum(pred * target) + ep
    union = torch.sum(pred) + torch.sum(target) + ep
    res = intersection / union
    return res
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    a = m2.max()
    intersection = (m1 * m2).sum().float()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def cal_IOU_seprate(pre,target):
    try:
        pre=pre[0].cpu().detach().numpy()
        target = target[0].cpu().detach().numpy()
    except:
        pre = pre[0]
        target = target[0]
    res_list = []
    for i in range(1,3):
        area_c = np.where((pre==target)&(pre ==i))
        area_a = np.where(pre==i)
        area_b = np.where(target==i)
        n_c = len(area_c[0])
        n_a =  len(area_a[0])
        n_b = len(area_b[0])
        iou = n_c/(n_a+n_b-n_c+0.001)
        res_list.append(np.round(iou,2))
    return res_list
def cal_dice_seprate(pred,target):
    dice_list= []
    class1 = pred[1]
    class2 = pred[2]

    dice1 = dice_coeff(class1,target[1])
    dice_list.append(round(dice1.item(),2))
    if target[2].sum()>20:
        dice2 = dice_coeff(class2,target[2])
        if dice2.item()>0.1:
            dice_list.append(round(dice2.item(),2))
    mean_dice = np.mean(dice_list)
    return dice_list,mean_dice
def cal_dice_seprate_mask(pred,target):
    pre = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    dice_list= []

    for i in range(1, 3):
        area_c = np.where((pre == target) & (pre == i))
        area_a = np.where(pre == i)
        area_b = np.where(target == i)
        n_c = len(area_c[0])
        n_a = len(area_a[0])
        n_b = len(area_b[0])
        if n_a>100:
            dice_score = (2*n_c+0.001)/(n_a+n_b+0.001)
        else:
            dice_score=0
        dice_list.append(np.round(dice_score,2))
    mean_dice = np.mean(dice_list)
    return dice_list,mean_dice

def cal_Haus_mask(pred,target):
    pre = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    Haus_score= 0
    pre_ = np.zeros(pre.shape,dtype=bool)
    target_ = np.zeros(target.shape, dtype=bool)
    target_[target==1]=True
    pre_[pre == 1] = True
    pos = np.where(target_==True)
    n = len(pos[0])

    surface_distances = surface_distance.compute_surface_distances(
        pre_, target_, spacing_mm=(1, 1))
    res95 = surface_distance.compute_robust_hausdorff(surface_distances, 95)
    Haus_score=np.round(res95,2)


    return Haus_score

def cal_Haus_seprate_mask(pred,target):
    pre = pred.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    Haus_list= []

    for i in range(1, 3):
        pre_ = np.zeros(pre.shape,dtype=bool)
        target_ = np.zeros(target.shape, dtype=bool)
        target_[target==i]=True
        pre_[pre == i] = True
        pos = np.where(target_==True)
        n = len(pos[0])
        if n>200:
            surface_distances = surface_distance.compute_surface_distances(
                pre_, target_, spacing_mm=(1, 1))
            res95 = surface_distance.compute_robust_hausdorff(surface_distances, 95)
            Haus_list.append(np.round(res95,2))
        else:
            Haus_list.append(0)


    return Haus_list
def check_outputs(val_inputs,val_labels,val_outputs,onehot=False,softmax=True):
    flag =0
    if softmax==True:
        val_outputs = torch.softmax(val_outputs,dim=1)
        mask = torch.argmax(val_outputs, axis=1)
        mask = mask[0].cpu().detach().numpy()
    else:
        mask = onehot2mask(val_outputs[0].cpu().detach().numpy())
    val_inputs_ = val_inputs.cpu().detach().numpy()

    label_ = val_labels[0].cpu().detach().numpy()
    val_outputs_ = val_outputs[0].cpu().detach().numpy()


    if onehot == True:
        mask_ = onehot2mask(label_)
    else:
        mask_ = label_

    if np.sum(mask_ == 2) != 0:
        flag = 1

    if flag != 0:
        plt.subplot(3, 3, 1)
        plt.title("input")
        plt.imshow(val_inputs_[0, 0])
        plt.subplot(3, 3, 2)
        plt.title("label")
        plt.imshow(mask_)
        plt.subplot(3, 3, 3)
        plt.title("prediction")
        plt.imshow(mask)
        for o in range(3):
            plt.subplot(3, 3, 4 + o)
            plt.imshow(val_outputs_[ o])
        labelx =mask2onehot(mask_)
        for o in range(3):
            plt.subplot(3, 3, 7 + o)
            plt.imshow(labelx[o])
        plt.show()
def plot_mask_label_img(image,mask,label,save_name,dice_list=[]):
    image = image*255
    plt.subplot(1,3,1)
    plt.imshow(image[0])
    plt.subplot(1,3,2)
    plt.imshow(mask)
    words = 'IOU class1,2: ' + str(dice_list[0]) + ',' + str(dice_list[1])
    plt.text(50, 50, words, bbox=dict(fill=False, edgecolor='red', linewidth=2))
    plt.subplot(1,3,3)
    plt.imshow(label)
    plt.savefig(save_name)

def plot_mask_label_img_signal(image,mask,label,save_name,dice_score=0):
    image = image*255
    plt.subplot(1,3,1)
    plt.imshow(image[0])
    plt.subplot(1,3,2)
    plt.imshow(mask)
    words = 'Dice: ' + str(dice_score)
    plt.text(50, 50, words, bbox=dict(fill=False, edgecolor='red', linewidth=2))
    plt.subplot(1,3,3)
    plt.imshow(label)
    plt.savefig(save_name)
def post_processing(mask,tensor=True,save_name = ''):
    if tensor ==True:
        before = mask.cpu().detach().numpy()
    else:
        before =mask

    if save_name != '':
        plt.figure(figsize=(8, 6))
        plt.subplot(2,3,1)
        plt.imshow(before[0])
        plt.title("Origin mask", fontsize=8)
    shape = before.shape

    for i in range(shape[0]):
        img_temp = before[i]
        new_img_temp = np.zeros((shape[1],shape[2]))
        mask1 = np.zeros(new_img_temp.shape,dtype=bool)
        mask2 = np.zeros(new_img_temp.shape, dtype=bool)
        mask1[img_temp==1]=1
        mask2[img_temp==2]=1
        if save_name != '':
            plt.subplot(2,3,2)
            plt.imshow(mask1)
            plt.title("Prediction Ureter",fontsize=8)
            plt.subplot(2, 3, 3)
            plt.imshow(mask2)
            plt.title("Prediction UA",fontsize=8)
        mask1 = morphology.remove_small_objects(mask1, 400)
        # mask1=morphology.binary_opening(mask1)
        mask2 = morphology.remove_small_objects(mask2, 400)
        # mask2 = morphology.binary_opening(mask2)
        if save_name != '':
            plt.subplot(2, 3, 4)
            plt.imshow(mask1)
            plt.title("After post-processing Ureter", fontsize=8)
            plt.subplot(2, 3, 5)
            plt.imshow(mask2)
            plt.title("After post-processing UA",fontsize=8)
        new_img_temp[mask1==True]=1
        new_img_temp[mask2==True]=2
        before[i]=new_img_temp
    if save_name != '':
        plt.subplot(2, 3, 6)
        plt.imshow(before[0])
        plt.title("Final result",fontsize=8)
        plt.savefig(save_name)

    if tensor == True:
        before = torch.tensor(before).cuda()

    return before

def plot_output(image,mask,save_name,dice_list=[]):
    image = image * 255
    image = np.moveaxis(image, 0, -1)
    mask = np.expand_dims(mask[0], 0).repeat(3, axis=0)
    mask = np.moveaxis(mask, 0, -1)
    New_mask1 = mask.copy()
    New_mask2 = mask.copy()
    New_mask1[:, :, 0] = 0
    New_mask1[:, :, 2] = 0

    New_mask2[:, :, 0] = 0
    New_mask2[:, :, 1] = 0
    image[New_mask1 == 1] = 255
    image[New_mask2 == 2] = 255
    image = np.flip(image, 2)
    if len(dice_list)!=0:
        if len(dice_list)==2:
            words = 'Dice UR,UA: ' + str(np.round(dice_list[0],2))+ ',' + str(np.round(dice_list[1],2))
        else:
            words = 'Dice UR: ' + str(np.round(dice_list[0],2))
        image=cv2.UMat(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(image, words, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(save_name, image)
    mask_name = save_name.replace("img",'mask')
    plt.figure(2)
    plt.imshow(mask[:,:,0])
    # plt.show()
    plt.savefig(mask_name)
def plot_evl_epochs(val_interval,epoch_loss_values,metric_values,save_same):
    plt.figure("train", (12, 10))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")



    plt.plot(x, y, color="purple")
    dicts = {'epoch_loss_values':epoch_loss_values,
             'metric_values':metric_values,
             }

    save_json = save_same.replace('png','json')
    with open(save_json, "w") as dump_f:
        json.dump(dicts, dump_f, indent=4)
    plt.figure(1)
    plt.savefig(save_same)
    plt.cla()
#=============================================================================
def multi_class_dice(pred, mask, fp_weight=1.0, label_smooth=1.0, eps=1e-6,
                     class_weight=None, ignore_index=None, per_instance=False):
    '''
    Multi class dice
    :param pred: [n_batch, n_class, ...] 1D to 3D inputs ranged in [0, 1] (as prob)
    :param mask: [n_batch, n_class, ...] 1D to 3D inputs as a 0/1 mask
    :param fp_weight: float [0,1], penalty for fp preds, may work in data with heavy fg/bg imbalances
    :param label_smooth: float (0, inf), power of the denominator
    :param eps: epsilon, avoiding zero-division
    :param class_weight: list [float], weights for classes
    :param ignore_index: int [0, n_class), num of classes to ignore; or list [int], indexes to ignore
    :param per_instance: boolean, if True, dice was calculated per instance instead of per batch
    :return: dice score.
    '''
    nd = len(pred.shape)
    n, c, *_ = pred.shape
    # assert nd in (3, 4, 5), 'Only support 3d to 5d tensors, got {}'.format(pred.shape)
    # assert pred.shape == mask.shape, 'Sizes of inputs do not match'
    i_d = 'ncijk'[:nd]
    o_d = 'nc' if per_instance else 'c'

    intersect = torch.einsum('{i_d}, {i_d} -> {o_d}'.format(i_d=i_d, o_d=o_d), pred, mask)
    sum_ = pred + mask if fp_weight == label_smooth == 1 else fp_weight * pred ** label_smooth + mask ** label_smooth
    union = torch.einsum('{i_d} -> {o_d}'.format(i_d=i_d, o_d=o_d), sum_)
    dice_ = (2 * intersect + eps) / (union + eps)

    if class_weight is None:
        dice = dice_
    else:
        # assert isinstance(class_weight, (list, tuple, torch.FloatTensor))
        if isinstance(class_weight, (list, tuple)):
            class_weight = torch.tensor(class_weight, dtype=torch.float32).to(dice_.device)
        if torch.max(class_weight) > 1:
            class_weight /= torch.sum(class_weight)
        dice = torch.einsum('...c, c -> c', dice_, class_weight)
    if ignore_index is None:
        return torch.mean(dice)
    else:
        # assert isinstance(ignore_index, (int, list, tuple))
        if isinstance(ignore_index, int):
            return torch.mean(dice[..., ignore_index:])
        else:
            select_dim = len(dice.shape) - 1
            select_index = [i for i in range(c) if i not in ignore_index]
            return torch.mean(torch.index_select(dice, select_dim, torch.tensor(select_index).to(dice.device)))


class BraTSDataset(Randomizable, CacheDataset):

    resource = {

    }
    md5 = {

    }

    def __init__(
            self,

            json_path: str,
            section: str,
            transform: Union[Sequence[Callable], Callable] = (),
            seed: int = 0,
            val_frac: float = 0.2,
            cache_num: int = sys.maxsize,
            cache_rate: float = 1.0,
            num_workers: int = 0,
    ) -> None:

        self.section = section
        self.val_frac = val_frac
        self.set_random_state(seed=seed)

        self.indices: np.ndarray = np.array([])
        data = self._generate_data_list(json_path)
        # as `release` key has typo in Task04 config file, ignore it.
        property_keys = [
            "description",
            "reference",
            "licence",
            "tensorImageSize",
            "modality",
            "numTraining",
            "numTest",
        ]
        self._properties = load_BraTS_datalist(json_path, property_keys)
        if transform == ():
            transform = LoadImaged(["t1", "t2", "t1ce", "flair", "seg"])
        super().__init__(data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)

    def get_indices(self) -> np.ndarray:
        """
        Get the indices of datalist used in this dataset.

        """
        return self.indices

    def randomize(self, data: List[int]) -> None:
        self.R.shuffle(data)

    def get_properties(self, keys: Optional[Union[Sequence[str], str]] = None):
        """
        Get the loaded properties of dataset with specified keys.
        If no keys specified, return all the loaded properties.

        """
        if keys is None:
            return self._properties
        elif self._properties is not None:
            return {key: self._properties[key] for key in ensure_tuple(keys)}
        else:
            return {}

    def _generate_data_list(self, json_path: str) -> List[Dict]:
        # section = "training" if self.section in ["training", "validation"] else "test"
        section = self.section
        datalist = load_BraTS_datalist(json_path, True, data_list_key=section)
        return self._split_datalist(datalist)

    def _split_datalist(self, datalist: List[Dict]) -> List[Dict]:
        if self.section == "test" or self.section == "validation" or self.section == "training":
            return datalist
        length = len(datalist)
        indices = np.arange(length)
        self.randomize(indices)

        val_length = int(length * self.val_frac)
        if self.section == "training":
            self.indices = indices[val_length:]
        else:
            self.indices = indices[:val_length]

        return [datalist[i] for i in self.indices]

class Resize_img(MapTransform):
    def __init__(self,  keys: KeysCollection,sizes):
        super(Resize_img, self).__init__(keys)
        # self.converter = AsChannelFirst(channel_dim=channel_dim)
        self.keys = keys
        self.size = sizes
    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for k in self.keys:
            if k != 'label':
                shape = d[k].shape
                img_pre = d[k]
                new_size = [self.size, self.size, shape[-1]]
                d[k] = np.zeros(new_size)
                img = transform.resize(img_pre, new_size, anti_aliasing=True, preserve_range=True)
                d[k] = img
            else:

                img_pre = d[k]
                new_size = [self.size, self.size]
                d[k] = np.zeros(new_size)
                img = transform.resize(img_pre, new_size,order=0,preserve_range=True)
                img[np.where((img>0.9)&(img<1.1))]=1
                img[np.where((img > 1.1) & (img < 1.9))] = 0
                img[img>1.9]=2
                img[img<0.9]=0
                d[k] = img

        return d

class NormalizeRGB(MapTransform):
    def __init__(self,  keys: KeysCollection,json_path):
        super(NormalizeRGB, self).__init__(keys)
        self.keys = keys
        self.json_path = json_path


    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)

        with open(self.json_path, 'r') as f:  # 读取json文件并返回data字典
            paras = json.load(f)
        means= paras['Means']
        stds = paras['Stds']
        for k in self.keys:
            data_p = d[k]
            for i in range(3):
                img_t = data_p[i]
                mean = means[i]
                std = stds[i]
                img_n = ((img_t/255.0)-mean)/std
                data_p[i]=img_n
            d[k]=data_p

        return d
class One_hot(MapTransform):
    def __init__(self, keys: KeysCollection, sizes=3):
        super(One_hot, self).__init__(keys)
        # self.converter = AsChannelFirst(channel_dim=channel_dim)
        self.keys = keys
        self.size = sizes

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        k = self.keys[0]
        label = d[k]
        shape = label.shape
        new_shape = [self.size,shape[0],shape[1]]
        d[k] = np.zeros(new_shape)
        one = mask2onehot(label, self.size)
        d[k]=one
        return d