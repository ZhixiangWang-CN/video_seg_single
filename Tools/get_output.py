import matplotlib.pyplot as plt
import numpy as np
import cv2
from Utils.utils import *
image = np.zeros((3,512,512),)
mask = np.zeros((512,512), np.uint8)
mask[200:300]=1
mask[300:400]=2
plot_output(image,mask,'image.png')