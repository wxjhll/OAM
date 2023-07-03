import torch
import numpy as np

milestones=np.linspace(10,100,10)
print(milestones)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,50], gamma=0.5)