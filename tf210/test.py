from model import unet
from make_dataset import *
import tensorflow as tf
from utils import *

batch_size=32
train_at, train_ping, val_at, val_ping=split_train_val(imgage_dir='D:/aDeskfile/OAM/AT',split=0.8)
train_data= get_dataset(image_dir=train_at, image_dir2=train_ping, is_shuffle=False, batch_size=batch_size)
train_data=train_data.prefetch(buffer_size=1)
val_data=get_dataset(image_dir=val_at, image_dir2=val_ping, is_shuffle=False, batch_size=1)

model=tf.keras.models.load_model('model.h5')
for x,y in val_data:
    pred=model(x)
    mae=np.sqrt((pred-y)**2)
    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    pre = axs[0, 0].imshow(pred[0],vmax=1,vmin=0,cmap='jet')
    fig.colorbar(pre, ax=axs[0, 0])
    axs[0, 0].set_title('pred')

    true = axs[0, 1].imshow(y[0],vmax=1,vmin=0,cmap='jet')
    fig.colorbar(true, ax=axs[0, 1])
    axs[0, 1].set_title('true')

    at = axs[1, 0].imshow(x[0],cmap='jet')
    fig.colorbar(at, ax=axs[1, 0])
    axs[1,0].set_title('at')

    mae = axs[1, 1].imshow(mae[0],vmax=1,vmin=0,cmap='jet')
    fig.colorbar(mae, ax=axs[1, 1])
    axs[1,1].set_title('mae')
    plt.show()





    '''
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(pred[0],cmap='jet')
    plt.clim(0,1)
    plt.title('pred')

    plt.subplot(2, 2, 2)
    plt.imshow(y[0],cmap='jet')
    plt.clim(0, 1)
    plt.title('true')

    plt.subplot(2, 2, 3)
    plt.imshow(x[0],cmap='jet')
    plt.title('at')

    plt.subplot(2, 2, 4)
    plt.imshow(mae[0],cmap='jet')
    plt.clim(0, 1)
    plt.title('gap')
    plt.show()
    '''