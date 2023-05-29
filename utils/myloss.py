import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
class my_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss1=torch.pow((x - y), 2)
        loss2=torch.abs(x-y)
        return torch.mean(loss1+loss2)