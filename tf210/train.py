from model import unet
from make_dataset import *
import tensorflow as tf
from util import *

batch_size=32
train_at, train_ping, val_at, val_ping=split_train_val(imgage_dir='D:/aDeskfile/OAM/AT',split=0.8)
train_dataset= get_dataset(image_dir=train_at, image_dir2=train_ping, is_shuffle=False, batch_size=batch_size)
train_dataset=train_dataset.prefetch(buffer_size=1)
val_data=get_dataset(image_dir=val_at, image_dir2=val_ping, is_shuffle=False, batch_size=32)

#回调
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./checkpoint',
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    save_freq=1)

model=unet(input_shape=(64,64,1))

model.summary()

STEPS_PER_EPOCH=12000/batch_size
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  initial_learning_rate=1e-3,
  decay_steps=STEPS_PER_EPOCH*100,
  decay_rate=1,
  staircase=False)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='MSE')
history=model.fit(train_dataset,validation_data=val_data,epochs=100,callbacks=None)
model.save('model.h5')
show_loss(history)