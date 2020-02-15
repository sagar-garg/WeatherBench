import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers

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

class ChannelSlice(tf.keras.layers.Layer):

    def __init__(self, n_out, **kwargs):
        self.n_out = n_out
        super().__init__(**kwargs)

    def _slice(self, inputs):
        # Input: [samples, lat, lon, filters]
        return inputs[..., :self.n_out]

    def __call__(self, inputs):
        out = Lambda(self._slice)(inputs)
        return out


def build_cnn(filters, kernels, input_shape, activation='elu', dr=0, periodic=True, l2=0):
    """Fully convolutional network"""
    x = input = Input(shape=input_shape)
    for f, k in zip(filters[:-1], kernels[:-1]):
        if periodic: x = PeriodicConv2D(f, k, activation=activation, kernel_regularizer=regularizers.l2(l2))(x)
        else: x = Conv2D(f, k, activation=activation, padding='same', kernel_regularizer=regularizers.l2(l2))(x)
        if dr > 0: x = Dropout(dr)(x)
    output = PeriodicConv2D(filters[-1], kernels[-1], kernel_regularizer=regularizers.l2(l2))(x)
    return keras.models.Model(input, output)

def resblock(inp, filters, kernel, negative_slope=0, bn_position=None, use_bias=True, res=True, n_conv=2, l2=0):
    x = inp
    for _ in range(n_conv):
        if bn_position == 'pre': x = BatchNormalization()(x)
        x = PeriodicConv2D(filters, kernel, use_bias=use_bias, kernel_regularizer=regularizers.l2(l2))(x)
        if bn_position == 'mid': x = BatchNormalization()(x)
        x = ReLU(negative_slope=negative_slope)(x)
        if bn_position == 'post': x = BatchNormalization()(x)
    if res: x = Add()([inp, x])
    return x

def build_resnet(filters, kernels, input_shape, negative_slope=0, bn_position=None, use_bias=True, l2=0):
    x = input = Input(shape=input_shape)
    x = PeriodicConv2D(filters[0], kernels[0], use_bias=use_bias, kernel_regularizer=regularizers.l2(l2))(x)
    x = ReLU(negative_slope=negative_slope)(x)
    for f, k in zip(filters[1:-1], kernels[1:-1]):
        x = resblock(x, f, k, negative_slope, bn_position, use_bias, l2=l2)
    output = PeriodicConv2D(filters[-1], kernels[-1], kernel_regularizer=regularizers.l2(l2))(x)
    return keras.models.Model(input, output)

