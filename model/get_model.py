from model.swin_unet import SwinUnet
from model.se_reunet import Shallow_SeResUNet,SeResUNet
from model.swin_unet_denoise import SUNet
from model.net3 import  net3
import torch.nn as nn


def load_model(model_name=''):
    if model_name=="swin_denoise":
        model = SUNet(img_size=128, patch_size=4, in_chans=1, out_chans=1,
                      embed_dim=96, depths=[2, 2, 2, 2],
                      num_heads=[8, 8, 8, 8],
                      window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                      drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                      norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                      use_checkpoint=False, final_upsample="Dual up-sample")
    if model_name == "swin_unet":
        model= SwinUnet(embed_dim=96,
                             patch_height=4,
                             patch_width=4,
                             class_num=1)
    if model_name == "seres_unet":
        model = SeResUNet(n_channels=1, n_classes=1, deep_supervision=False,
                            dropout=False, rate=0.3)

    return model
