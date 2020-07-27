import tensorflow as tf  # I am using tensorflow=2.1
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]=str(3)


# For me it is important the the convolutions are periodic in y
# In x I am simply using zero padding.
class PeriodicPadding2D(tf.keras.layers.Layer):
    def __init__(self, pad_width, **kwargs):
        super().__init__(**kwargs)
        self.pad_width = pad_width

    def call(self, inputs, **kwargs):
        if self.pad_width == 0:
            return inputs
        inputs_padded = tf.concat(
            [inputs[:, :, -self.pad_width:, :], inputs, inputs[:, :, :self.pad_width, :]], axis=2)
        # Zero padding in the lat direction
        inputs_padded = tf.pad(inputs_padded, [[0, 0], [self.pad_width, self.pad_width], [0, 0], [0, 0]])
        return inputs_padded

    def get_config(self):
        config = super().get_config()
        config.update({'pad_width': self.pad_width})
        return config

class PeriodicConv2D(tf.keras.layers.Layer):
    def __init__(self, filters,
                 kernel_size,
                 conv_kwargs={},
                 **kwargs, ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv_kwargs = conv_kwargs
        if type(kernel_size) is not int:
            assert kernel_size[0] == kernel_size[1], 'PeriodicConv2D only works for square kernels'
            kernel_size = kernel_size[0]
        pad_width = (kernel_size - 1) // 2
        self.padding = PeriodicPadding2D(pad_width)
        self.conv = Conv2D(
            filters, kernel_size, padding='valid', **conv_kwargs
        )

    def call(self, inputs):
        return self.conv(self.padding(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'conv_kwargs': self.conv_kwargs})
        return config

def convblock(inputs, filters, kernel=3, stride=1, bn_position=None, l2=0,
              use_bias=True, dropout=0, activation='relu'):
    x = inputs
    if bn_position == 'pre': x = BatchNormalization()(x)
    x = PeriodicConv2D(
        filters, kernel, conv_kwargs={
            'kernel_regularizer': regularizers.l2(l2),
            'use_bias': use_bias
        }
    )(x)
    if bn_position == 'mid': x = BatchNormalization()(x)
    x = LeakyReLU()(x) if activation == 'leakyrelu' else Activation(activation)(x)
    if bn_position == 'post': x = BatchNormalization()(x)
    if dropout > 0: x = Dropout(dropout)(x)
    return x

def resblock(inputs, filters, kernel, bn_position=None, l2=0, use_bias=True,
             dropout=0, skip=True, activation='relu', down=False, up=False):
    x = inputs
    if down:
        x = MaxPooling2D()(x)
    for i in range(2):
        x = convblock(
            x, filters, kernel, bn_position=bn_position, l2=l2, use_bias=use_bias,
            dropout=dropout, activation=activation
        )
    if down or up:
        inputs = PeriodicConv2D(
            filters, kernel, conv_kwargs={
                'kernel_regularizer': regularizers.l2(l2),
                'use_bias': use_bias,
                'strides': 2 if down else 1
            }
        )(inputs)
    if skip: x = Add()([inputs, x])
    return x

def build_resnet(filters, kernels, input_shape, bn_position=None, use_bias=True, l2=0,
                 skip=True, dropout=0, activation='relu', long_skip=False,
                 **kwargs):
    x = input = Input(shape=input_shape)

    # First conv block to get up to shape
    x = ls = convblock(
        x, filters[0], kernels[0], bn_position=bn_position, l2=l2, use_bias=use_bias,
        dropout=dropout, activation=activation
    )

    # Resblocks
    for f, k in zip(filters[1:-1], kernels[1:-1]):
        x = resblock(x, f, k, bn_position=bn_position, l2=l2, use_bias=use_bias,
                dropout=dropout, skip=skip, activation=activation)
        if long_skip:
            x = Add()([x, ls])

    # Final convolution
    output = PeriodicConv2D(
        filters[-1], kernels[-1],
        conv_kwargs={'kernel_regularizer': regularizers.l2(l2)},
    )(x)

    # This is just because I am using mixed precision. Can be left out for regular precision.
    output = Activation('linear', dtype='float32')(output)
    return keras.models.Model(input, output)


# Load the example batches
# These are the paths for servus03
X = np.load('/data/stephan/tmp/sample_X.npy')
y = np.load('/data/stephan/tmp/sample_y.npy')

# Define the model
# Usually I have a lot more layers
model = build_resnet(
    filters=[128, 128, 128, 2],
    kernels=[7, 3, 3, 3],
    input_shape=(32, 64, 114,),
    bn_position='post',
    dropout=0.1,   # I am currently using a combination of dropout and l2 for regularization
    l2=1e-5,       # Of course it would be great if I didn't have to use them with racecar
    activation='leakyrelu',
)

model.compile('adam', 'mse')
pred = model.predict(X)
print(pred.shape)