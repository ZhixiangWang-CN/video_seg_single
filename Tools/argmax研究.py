import torch
import matplotlib.pyplot as plt

data = torch.zeros((3,5,5))
data[2]=0.9
data[1][2:4,2:4]=0.95
data[0]=0.7
res = torch.argmax(data,dim=0)
print(res.shape)
plt.imshow(res)
plt.show()