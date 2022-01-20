import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology

img = np.zeros((64,64))

img[20:30,20:30]=1
img[30:35,30:35]=2
img_1 = np.zeros((64,64),dtype=bool)
img_2 = np.zeros((64,64),dtype=bool)
img_1[img==1]=1
img_2[img==2]=1
plt.subplot(1,3,1)
plt.imshow(img)

after1 = morphology.remove_small_objects(img_1,64)
plt.subplot(1,3,2)
plt.imshow(after1)
after2 = morphology.remove_small_objects(img_2,64)
plt.subplot(1,3,3)
plt.imshow(after2)
plt.show()