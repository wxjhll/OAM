import  numpy as np
import glob
import pathlib
from PIL import Image
from tqdm import tqdm
def get_image_paths(image_dir=None):
    data_path = pathlib.Path(image_dir)
    paths = list(data_path.glob('*'))
    paths = [str(p) for p in paths]
    #train_data_path = glob.glob( 'E:/data/train/*/*.jpg' )
    return paths

paths=get_image_paths(image_dir='D:/aDeskfile/OAM/AT')
mean=0
var=0
max=0
for path in tqdm(paths):
    image=Image.open(path)
    image=np.asarray(image)/255.
    #mean+=np.mean(image)
    var+=np.mean((image-0.12622979)**2)
    #print(np.mean(image))


#mean /= len(paths) #0.16436058168598208
#print(mean)
var/=len(paths)
print(np.sqrt(var))#0.0334704598535126
