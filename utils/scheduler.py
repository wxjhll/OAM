import torch
import numpy as np
from torch.optim import lr_scheduler

def get_scheduler(optimizer, cfg):
    if cfg== 'CosLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=50,
                                                   eta_min=1e-6)
    elif cfg == 'CosWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=10,
                                                             T_mult=1,
                                                             eta_min=1e-6,)
    elif cfg== 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=cfg.min_lr, )
    elif cfg== 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    elif cfg=='OneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                                        total_steps=100,
                                                        pct_start=0.3,
                                                        div_factor=10,
                                                        final_div_factor=1,
                                                        anneal_strategy='cos')

    elif cfg=='StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=5,
                                                    gamma=0.1)
    else:
        scheduler = None

    return scheduler