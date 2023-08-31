from unet import TransUnet
import torch
import time
a = torch.rand(2, 1, 128, 128)

model = TransUnet(in_channels=1, img_dim=128, vit_blocks=1, vit_dim_linear_mhsa_block=512,
                  classes=1)
y = model(a)

now = int(time.time())
#转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
timeArray = time.localtime(now)
otherStyleTime = time.strftime("%m-%d/%H:%M", timeArray)
print(otherStyleTime)