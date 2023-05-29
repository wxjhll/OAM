from model import unet
from make_dataset import *
import tensorflow as tf
from utils import *
import keras
batch_size=32


train_at, train_ping, val_at, val_ping=split_train_val(imgage_dir='D:/aDeskfile/OAM/AT',split=0.8)
train_dataset= get_dataset(image_dir=train_at, image_dir2=train_ping, is_shuffle=False, batch_size=batch_size)
train_dataset=train_dataset.prefetch(buffer_size=1)

val_data=get_dataset(image_dir=val_at, image_dir2=val_ping, is_shuffle=False, batch_size=32)

#回调
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='best.h5',
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    )

#model=unet(input_shape=(64,64,1))
model = tf.keras.models.Sequential([
    # 对模型做归一化的处理，将0-255之间的数字统一处理到0到1之间
    tf.keras.layers.Input(shape=(64,64,1)),
    # 卷积层，该卷积层的输出为32个通道，卷积核的大小是3*3，激活函数为relu
    tf.keras.layers.Conv2D(32,(5,5),padding='same',kernel_regularizer=keras.regularizers.l2(1e-3)),
    tf.keras.layers.LeakyReLU(alpha=0.3),
#,kernel_regularizer=keras.regularizers.l2(1e-4)
    tf.keras.layers.Conv2D(32, (3, 3),padding='same',kernel_regularizer=keras.regularizers.l2(1e-4)),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.MaxPooling2D((2, 2),padding='same'),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.MaxPooling2D((2, 2),padding='same'),

    tf.keras.layers.Conv2D(64, (3, 3),padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.MaxPooling2D((2, 2),padding='same'),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.3),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.3),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same',),
    tf.keras.layers.LeakyReLU(alpha=0.3),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.3),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same',),
    tf.keras.layers.LeakyReLU(alpha=0.3),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.3),

    tf.keras.layers.UpSampling2D((2,2),interpolation='bilinear'),
    tf.keras.layers.Conv2D(64,(3, 3),padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.3),
#
    tf.keras.layers.UpSampling2D((2,2),interpolation='bilinear'),
    tf.keras.layers.Conv2D(32, (3, 3),padding='same'),
    tf.keras.layers.LeakyReLU(alpha=0.3),

    tf.keras.layers.UpSampling2D((2, 2),interpolation='bilinear'),
    tf.keras.layers.Conv2D(16, (3, 3), padding='same',kernel_regularizer=keras.regularizers.l2(1e-3),),
    tf.keras.layers.LeakyReLU(alpha=0.3),

    tf.keras.layers.Conv2D(16, (3, 3),padding='same',kernel_regularizer=keras.regularizers.l2(1e-4),),
    tf.keras.layers.LeakyReLU(alpha=0.3),

    # tf.keras.layers.Conv2D(16, (3, 3),padding='same',kernel_regularizer=keras.regularizers.l2(1e-5)),
    # tf.keras.layers.LeakyReLU(alpha=0.3),

    tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid',padding='same'),

])
model.summary()

STEPS_PER_EPOCH=27000/batch_size
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  initial_learning_rate=1e-3,
  decay_steps=STEPS_PER_EPOCH*100,
  decay_rate=1,
  staircase=False)

#训练
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='MSE')
history=model.fit(train_dataset,validation_data=val_data,epochs=100,callbacks=[model_checkpoint_callback])
model.save('model.h5')
show_loss(history)