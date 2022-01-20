import torch
from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from  model.load_model import *

feature_number = 0
save_name = 'a'
def get_model(name = '3DUNet'):
    path = '../training_results/' + name + '/'
    model = load_model(name='')
    model.load_state_dict(
        torch.load(path + '/best_metric_model.pth')
    )
    print(model)

    model.eval()
    model.cuda()
    return model
def viz(module, input):
    global feature_number
    global save_name
    x = input[0][0]
    #最多显示4张图
    min_num = np.minimum(4, x.size()[0])

    for i in range(min_num):
        plt.subplot(1, 4, i+1)
        plt.imshow(x[i].cpu())
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
    # plt.show()
    # plt.title(save_name)
    plt.savefig("./features_map/"+str(feature_number)+'.tif')
    print(feature_number)
    feature_number+=1
    plt.clf()

import cv2
import numpy as np
def main():
    t = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((512, 512)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                            ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model =get_model()
    # model = resnet18(pretrained=True).to(device)
    for name, m in model.named_modules():
    # for name,m in model.state_dict():

        print(name)
        # if not isinstance(m, torch.nn.ModuleList) and \
        #         not isinstance(m, torch.nn.Sequential) and \
        #         type(m) in torch.nn.__dict__.values():
        # 这里只对卷积层的feature map进行显示
        if isinstance(m, torch.nn.Conv2d):
            global save_name
            print("===",name)
            m.register_forward_pre_hook(viz)
    img = Image.open("C:/Dataset/video/Segdataset/UR/image/2260aimg.png")
    img = np.array(img)
    img = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(img)

if __name__ == '__main__':
    main()