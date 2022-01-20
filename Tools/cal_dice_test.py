import torch

from Utils.utils import *

label = torch.zeros(4,3,64,64,64)
pre = torch.zeros(4,3,64,64,64)
label[:,:,20:40,20:40,20:40]=1
pre[:,:,10:40,10:40,10:40]=1

res = cal_dice(pre,label)
print(res)