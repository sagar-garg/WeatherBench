import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *

def limit_mem():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)


class PeriodicConv2D(tf.keras.layers.Conv2D):
    """Convolution with periodic padding in second spatial dimension (lon)"""

    def __init__(self, filters, kernel_size, **kwargs):
        assert type(kernel_size) is int, 'Periodic convolutions only works for square kernels.'
        self.pad_width = (kernel_size - 1) // 2
        super().__init__(filters, kernel_size, **kwargs)
        assert self.padding == 'valid', 'Periodic convolution only works for valid padding.'
        assert sum(self.strides) == 2, 'Periodic padding only works for stride (1, 1)'

    def _pad(self, inputs):
        # Input: [samples, lat, lon, filters]
        # Periodic padding in lon direction
        inputs_padded = tf.concat(
            [inputs[:, :, -self.pad_width:, :], inputs, inputs[:, :, :self.pad_width, :]], axis=2)
        # Zero padding in the lat direction
        inputs_padded = tf.pad(inputs_padded, [[0, 0], [self.pad_width, self.pad_width], [0, 0], [0, 0]])
        return inputs_padded

    def __call__(self, inputs, *args, **kwargs):
        # Unfortunate workaround necessary for TF < 1.13
        inputs_padded = Lambda(self._pad)(inputs)
        return super().__call__(inputs_padded, *args, **kwargs)

def build_cnn(filters, kernels, input_shape, activation='elu', dr=0):
    """Fully convolutional network"""
    x = input = Input(shape=input_shape)
    for f, k in zip(filters[:-1], kernels[:-1]):
        x = PeriodicConv2D(f, k, activation=activation)(x)
        if dr > 0: x = Dropout(dr)(x)
    output = PeriodicConv2D(filters[-1], kernels[-1])(x)
    return keras.models.Model(input, output)

