import numpy as np
import matplotlib.pyplot as plt
label_p = "C:/Dataset/video/Segdataset/UR/mask/1902amask.npy"

data = np.load(label_p)
data[data==2]=0
plt.imshow(data)
plt.show()