import numpy as np
from torchvision import datasets, models, transforms
from data.make_dataset import *
from model.se_reunet import Shallow_SeResUNet
import torch.nn.functional as F
from PIL import Image

transform = transforms.Compose([
transforms.ToTensor(),
transforms.Resize([128,128]),
transforms.Normalize(mean=[0.5], std=[0.5])])

def cnn_ping(at_dir='at.png'):
    at_img =Image.open(at_dir)
    at_img=at_img.convert('L')
    at_img = transform(at_img)
    at_img=at_img.unsqueeze(dim=0)
    net = Shallow_SeResUNet(n_channels=1, n_classes=1, deep_supervision=False,
                            dropout=False, rate=0.1)
    net.load_state_dict(torch.load('../weight/best{}.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    with torch.no_grad():
        at_img=at_img.to(device)
        outputs = net(at_img)

        ping=outputs[0].cpu().squeeze()
        ping=ping.numpy()
        # ping_2pi=ping*2*np.pi

        # mask_ping=np.zeros((512,512))
        # mask_ping[96:416,96:416]=ping
    return ping

if __name__ == '__main__':
    cnn_ping(at_dir='origin_at0.png')

