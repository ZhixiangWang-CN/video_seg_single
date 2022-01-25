from monai.losses import *
from torch.nn import *

import torch
import torch.nn as nn


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_cross_entory = CrossEntropyLoss()
        self.loss_Dice = DiceLoss(to_onehot_y=False, sigmoid=False, squared_pred=True)
        self.loss_function_focal = FocalLoss()
        self.Dce = DiceCELoss()
        self.loss_BCE = nn.BCELoss()
        self.loss_tky = TverskyLoss(beta=0.9, alpha=0.1)

    def forward(self, x, y):


        # y = torch.tensor(y, dtype=torch.float)
        # BCE= self.loss_BCE(x,y)
        # fc=self.loss_function_focal(x,y)
        # DCE = self.Dce(x,y)
        #=====================

        # CE = self.loss_cross_entory(x,y)
        x = x[:,0,:,:]
        BCE = self.loss_BCE(x, y)
        # x = torch.softmax(x,dim=1)
        # x = torch.argmax(x,dim=1)
        # x= x[:,0,:,:]
        Dice = self.loss_Dice(x, y)
        # #===============================
        # HD = self.HD_loss(x,y)
        # norm_HD = (1-torch.exp(-1*HD))/(1+torch.exp(-1*HD))
        #===============================
        return BCE+Dice

