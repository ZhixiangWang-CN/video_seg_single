import numpy as np
import matplotlib.pyplot as plt
from Utils.utils import *
data = np.load('C:/Dataset/video/Segdataset/UR/mask/000423mask.npy')

one = mask2onehot(data,3)
print(one.shape)
for i in range(3):
    plt.subplot(2,2,1+i)
    plt.imshow(one[i])
con = onehot2mask(one)
plt.subplot(2,2,4)
plt.imshow(con)
plt.show()