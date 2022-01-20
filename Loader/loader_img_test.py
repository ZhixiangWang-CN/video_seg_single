from monai.apps import DecathlonDataset
from Utils.utils import BraTSDataset
from Utils.utils import *
from monai.transforms import *
from monai.data import DataLoader
import matplotlib.pyplot as plt
import scipy
from PIL import Image

def read_data_ds(data_path='../Tools/dataset.json',new_size=512,cache_num=2):

    test_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys= ['image',"label"]),
            Resize_img(keys= ['image',"label"],sizes=new_size),
            # One_hot(keys= ["label"]),
            AsChannelFirstd(keys=['image'], channel_dim=-1),
            NormalizeRGB(keys=['image'],json_path='C:/Softwares/Codes/video_seg/Tools/Mean_std.json'),

            ToTensord(keys=['image',"label"]),
        ]
    )

    test_ds = BraTSDataset(
        json_path=data_path,
        transform=test_transform,
        section="validation",
        num_workers=0,
        cache_num=cache_num,


    )


    return test_ds


if __name__ == '__main__':
    # path = 'C:/Softwares/Codes/BrainMRI/BrainMRISegmentation/utils/shuffle_UNet.json'
    test_ds=read_data_ds()
    val_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    train_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    for epoch in range(3):
        for batch_data in val_loader:
            # for i in range(4):
            # plt.subplot(1,3,1)
            img = batch_data["image"][0].detach().cpu().numpy()
            a = img.mean()
            # print("after=",a)
            img = np.moveaxis(img,0,-1)
            # im_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # # plt.imshow(img)
            # im = Image.fromarray(img)
            # im.show()
            # cv2.imshow('a',im_rgb)
            # cv2.waitKey()
            # plt.subplot(1,3,2)
            # label = batch_data["label"][0].detach().cpu().numpy()
            # label[label==1]=0
            # plt.imshow(label)
            # plt.subplot(1, 3, 3)
            # plt.imshow(img+label)
            # plt.show()
            # break
