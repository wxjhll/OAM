import tensorflow as tf


def conv_block(inputs, filters, kernel_size=3):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding="same")(inputs)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding="same")(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def unet(input_shape=(128,128,1), num_classes=1):
    # 编码器
    inputs = tf.keras.layers.Input(input_shape)
    conv1 = conv_block(inputs, 32, kernel_size=3)

    pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 64, kernel_size=3)

    pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 128, kernel_size=3)

    pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 256, kernel_size=3)


    # 解码器
    up6 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv4)
    concat6 = tf.keras.layers.concatenate([up6, conv3], axis=-1)
    conv6 = conv_block(concat6, 128, kernel_size=3)

    up7 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv6)
    concat7 = tf.keras.layers.concatenate([up7, conv2], axis=-1)
    conv7 = conv_block(concat7, 64, kernel_size=3)

    up8 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv7)
    concat8 = tf.keras.layers.concatenate([up8, conv1], axis=-1)
    conv8 = conv_block(concat8, 32, kernel_size=3)


    outputs = tf.keras.layers.Conv2D(1, kernel_size=1, activation="sigmoid")(conv8)

    model = tf.keras.models.Model(inputs, outputs)

    return model