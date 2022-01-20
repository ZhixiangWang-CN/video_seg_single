import numpy as np
import surface_distance

pre = np.zeros((64,64),dtype=bool)
label= np.zeros((64,64),dtype=bool)
pre[20:40,20:40]=True
pre[60,60]=True
label[30:40,30:40]=True

surface_distances = surface_distance.compute_surface_distances(
    pre, label, spacing_mm=(1,1))
res95 = surface_distance.compute_robust_hausdorff(surface_distances, 95)
res100 = surface_distance.compute_robust_hausdorff(surface_distances, 100)
dice = surface_distance.compute_dice_coefficient(label,pre)
print(res95,res100)
print(dice)