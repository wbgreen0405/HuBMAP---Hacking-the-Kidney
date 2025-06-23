import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, LeakyReLU, Dropout,
                                     MaxPooling2D, Conv2DTranspose, concatenate)
from tensorflow.keras.models import Model

# --- Basic UNet++ style encoder-decoder, can be expanded per your original code ---

def conv2d(filters):
    return Conv2D(filters=filters, kernel_size=(3, 3), padding='same')

def conv2dtranspose(filters):
    return Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same')

def UNetPP(input_shape=(256, 256, 3), num_classes=1, number_of_filters=2, dropout_rate=0.2):
    """A simplified UNet++ implementation suitable for HuBMAP patch-based segmentation."""
    model_input = Input(input_shape)

    # Encoder
    x00 = conv2d(16 * number_of_filters)(model_input)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    x00 = Dropout(dropout_rate)(x00)
    x00 = conv2d(16 * number_of_filters)(x00)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    x00 = Dropout(dropout_rate)(x00)
    p0 = MaxPooling2D(pool_size=(2, 2))(x00)

    x10 = conv2d(32 * number_of_filters)(p0)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    x10 = Dropout(dropout_rate)(x10)
    x10 = conv2d(32 * number_of_filters)(x10)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    x10 = Dropout(dropout_rate)(x10)
    p1 = MaxPooling2D(pool_size=(2, 2))(x10)

    x20 = conv2d(64 * number_of_filters)(p1)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    x20 = Dropout(dropout_rate)(x20)
    x20 = conv2d(64 * number_of_filters)(x20)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    x20 = Dropout(dropout_rate)(x20)
    p2 = MaxPooling2D(pool_size=(2, 2))(x20)

    x30 = conv2d(128 * number_of_filters)(p2)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    x30 = Dropout(dropout_rate)(x30)
    x30 = conv2d(128 * number_of_filters)(x30)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    x30 = Dropout(dropout_rate)(x30)
    p3 = MaxPooling2D(pool_size=(2, 2))(x30)

    m = conv2d(256 * number_of_filters)(p3)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    m = Dropout(dropout_rate)(m)
    m = conv2d(256 * number_of_filters)(m)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    m = Dropout(dropout_rate)(m)

    # Decoder (skip connections for UNet++)
    x31 = conv2dtranspose(128 * number_of_filters)(m)
    x31 = concatenate([x31, x30])
    x31 = conv2d(128 * number_of_filters)(x31)
    x31 = BatchNormalization()(x31)
    x31 = LeakyReLU(0.01)(x31)
    x31 = Dropout(dropout_rate)(x31)

    x22 = conv2dtranspose(64 * number_of_filters)(x31)
    x22 = concatenate([x22, x20])
    x22 = conv2d(64 * number_of_filters)(x22)
    x22 = BatchNormalization()(x22)
    x22 = LeakyReLU(0.01)(x22)
    x22 = Dropout(dropout_rate)(x22)

    x13 = conv2dtranspose(32 * number_of_filters)(x22)
    x13 = concatenate([x13, x10])
    x13 = conv2d(32 * number_of_filters)(x13)
    x13 = BatchNormalization()(x13)
    x13 = LeakyReLU(0.01)(x13)
    x13 = Dropout(dropout_rate)(x13)

    x04 = conv2dtranspose(16 * number_of_filters)(x13)
    x04 = concatenate([x04, x00])
    x04 = conv2d(16 * number_of_filters)(x04)
    x04 = BatchNormalization()(x04)
    x04 = LeakyReLU(0.01)(x04)
    x04 = Dropout(dropout_rate)(x04)

    # Final segmentation layer
    output = Conv2D(num_classes, kernel_size=(1, 1), activation='sigmoid')(x04)

    model = Model(inputs=[model_input], outputs=[output])
    return model

# Example usage:
# model = UNetPP(input_shape=(256,256,3), num_classes=1)
