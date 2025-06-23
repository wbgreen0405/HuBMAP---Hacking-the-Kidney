import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, Dropout,
    MaxPooling2D, Conv2DTranspose, concatenate
)
from tensorflow.keras.models import Model

def conv2d(filters, kernel_size=3, activation=True, dropout_rate=0.2):
    def layer(x):
        x = Conv2D(filters, (kernel_size, kernel_size), padding='same')(x)
        x = BatchNormalization()(x)
        if activation:
            x = LeakyReLU(0.01)(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        return x
    return layer

def conv2dtranspose(filters, kernel_size=2):
    def layer(x):
        return Conv2DTranspose(filters, (kernel_size, kernel_size), strides=(2, 2), padding='same')(x)
    return layer

def UNetPP(input_shape=(256, 256, 3), num_classes=1, number_of_filters=2, dropout_rate=0.2):
    """UNet++ implementation (nested skip connections, 4 levels deep, patch size 256x256)."""
    inputs = Input(input_shape)

    # Encoder
    x00 = conv2d(16 * number_of_filters, dropout_rate=dropout_rate)(inputs)
    x00 = conv2d(16 * number_of_filters, dropout_rate=dropout_rate)(x00)
    p0 = MaxPooling2D(pool_size=(2, 2))(x00)

    x10 = conv2d(32 * number_of_filters, dropout_rate=dropout_rate)(p0)
    x10 = conv2d(32 * number_of_filters, dropout_rate=dropout_rate)(x10)
    p1 = MaxPooling2D(pool_size=(2, 2))(x10)

    x20 = conv2d(64 * number_of_filters, dropout_rate=dropout_rate)(p1)
    x20 = conv2d(64 * number_of_filters, dropout_rate=dropout_rate)(x20)
    p2 = MaxPooling2D(pool_size=(2, 2))(x20)

    x30 = conv2d(128 * number_of_filters, dropout_rate=dropout_rate)(p2)
    x30 = conv2d(128 * number_of_filters, dropout_rate=dropout_rate)(x30)
    p3 = MaxPooling2D(pool_size=(2, 2))(x30)

    m = conv2d(256 * number_of_filters, dropout_rate=dropout_rate)(p3)
    m = conv2d(256 * number_of_filters, dropout_rate=dropout_rate)(m)

    # Decoder (nested skip connections)
    x31 = conv2dtranspose(128 * number_of_filters)(m)
    x31 = concatenate([x31, x30])
    x31 = conv2d(128 * number_of_filters, dropout_rate=dropout_rate)(x31)
    x31 = conv2d(128 * number_of_filters, dropout_rate=dropout_rate)(x31)

    x22 = conv2dtranspose(64 * number_of_filters)(x31)
    x22 = concatenate([x22, x20])
    x22 = conv2d(64 * number_of_filters, dropout_rate=dropout_rate)(x22)
    x22 = conv2d(64 * number_of_filters, dropout_rate=dropout_rate)(x22)

    x13 = conv2dtranspose(32 * number_of_filters)(x22)
    x13 = concatenate([x13, x10])
    x13 = conv2d(32 * number_of_filters, dropout_rate=dropout_rate)(x13)
    x13 = conv2d(32 * number_of_filters, dropout_rate=dropout_rate)(x13)

    x04 = conv2dtranspose(16 * number_of_filters)(x13)
    x04 = concatenate([x04, x00])
    x04 = conv2d(16 * number_of_filters, dropout_rate=dropout_rate)(x04)
    x04 = conv2d(16 * number_of_filters, dropout_rate=dropout_rate)(x04)

    output = Conv2D(num_classes, kernel_size=(1, 1), activation='sigmoid')(x04)

    model = Model(inputs=inputs, outputs=output)
    return model
