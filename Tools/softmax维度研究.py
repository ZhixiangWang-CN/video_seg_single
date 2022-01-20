import matplotlib.pyplot as plt
import torch

a = torch.zeros((3,3,20,20))
# print(a)
a[:,2,:,:]=1
b = torch.softmax(a,dim=1)
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(b[0,i])
plt.show()
print(b)