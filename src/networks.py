import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import *
from tensorflow.keras import regularizers
import numpy as np
#tf.enable_eager_execution() #added. so as to be able to use numpy arrays easily
def limit_mem():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.compat.v1.InteractiveSession(config=config)


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
    x = Activation(activation)(x)
    if bn_position == 'post': x = BatchNormalization()(x)
    if dropout > 0: x = Dropout(dropout)(x)
    return x

def resblock(inputs, filters, kernel, bn_position=None, l2=0, use_bias=True,
             dropout=0, skip=True):
    x = inputs
    for _ in range(2):
        x = convblock(
            x, filters, kernel, bn_position=bn_position, l2=l2, use_bias=use_bias,
            dropout=dropout
        )
    if skip: x = Add()([inputs, x])
    return x



def build_resnet(filters, kernels, input_shape, bn_position=None, use_bias=True, l2=0,
                 skip=True, dropout=0):
    x = input = Input(shape=input_shape)

    # First conv block to get up to shape
    x = convblock(
        x, filters[0], kernels[0], bn_position=bn_position, l2=l2, use_bias=use_bias,
        dropout=dropout
    )

    # Resblocks
    for f, k in zip(filters[1:-1], kernels[1:-1]):
        x = resblock(x, f, k, bn_position=bn_position, l2=l2, use_bias=use_bias,
                dropout=dropout, skip=skip)

    # Final convolution
    output = PeriodicConv2D(
        filters[-1], kernels[-1], conv_kwargs={'kernel_regularizer': regularizers.l2(l2)})(x)
    return keras.models.Model(input, output)


def build_unet(input_shape, n_layers, filters_start, channels_out, kernel=3, u_skip=True,
               res_skip=True, l2=0, bn_position=None, dropout=0):
    "https://github.com/Nishanksingla/UNet-with-ResBlock/blob/master/resnet34_unet_model.py"
    x = input = Input(shape=input_shape)
    filters = filters_start

    # Down
    down_layers = []
    for i in range(n_layers):
        # Resblock
        x_res = PeriodicConv2D(
            filters, 1, conv_kwargs={
                'use_bias': False, 'kernel_regularizer': regularizers.l2(l2)})(x)
        x = convblock(x, filters, kernel, bn_position=bn_position, l2=l2, dropout=dropout)
        x = convblock(x, filters, kernel, bn_position=bn_position, l2=l2, activation='linear',
                      dropout=dropout)
        if res_skip: x = Add()([x, x_res])
        x = ReLU()(x)
        if not i == n_layers - 1:
            down_layers.append(x)
            # Downsampling
            x = MaxPooling2D()(x)
            filters *= 2

    # Up
    for dl in reversed(down_layers):
        filters //= 2
        # Upsample
        x = UpSampling2D()(x)
        x = PeriodicConv2D(filters, 3, conv_kwargs={'kernel_regularizer': regularizers.l2(l2)})(x)
        x = ReLU()(x)

        # Concatenate
        if u_skip:
            x = Concatenate()([x, dl])

        # Resblock
        x_res = PeriodicConv2D(filters, 1, conv_kwargs={'use_bias': False})(x)
        x = convblock(x, filters, kernel, bn_position=bn_position, l2=l2, dropout=dropout)
        x = convblock(x, filters, kernel, bn_position=bn_position, l2=l2, activation='linear',
                      dropout=dropout)
        if res_skip: x = Add()([x, x_res])
        x = ReLU()(x)

    x = PeriodicConv2D(channels_out, 1, conv_kwargs={'kernel_regularizer': regularizers.l2(l2)})(x)
    return keras.models.Model(input, x)


def create_lat_mse(lat):
    weights_lat = np.cos(np.deg2rad(lat)).values
    weights_lat /= weights_lat.mean()
    def lat_mse(y_true, y_pred):
        error = y_true - y_pred
        mse = (error)**2 * weights_lat[None, : , None, None]
        return mse
    return lat_mse


### Scher
def build_scher_model(conv_depth, kernel_size, hidden_size, n_hidden_layers, lr):
    model = keras.Sequential([

                                 ## Convolution with dimensionality reduction (similar to Encoder in an autoencoder)
                                 Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation,
                                               input_shape=(Nlat, Nlon, n_channels)),
                                 layers.MaxPooling2D(pool_size=pool_size),
                                 Dropout(drop_prob),
                                 Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
                                 layers.MaxPooling2D(pool_size=pool_size),
                                 # end "encoder"

                                 # dense layers (flattening and reshaping happens automatically)
                             ] + [layers.Dense(hidden_size, activation='sigmoid') for i in range(n_hidden_layers)] +

                             [

                                 # start "Decoder" (mirror of the encoder above)
                                 Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
                                 layers.UpSampling2D(size=pool_size),
                                 Convolution2D(conv_depth, kernel_size, padding='same', activation=conv_activation),
                                 layers.UpSampling2D(size=pool_size),
                                 layers.Convolution2D(n_channels, kernel_size, padding='same', activation=None)
                             ]
                             )

    optimizer = keras.optimizers.adam(lr=lr)

    if N_gpu > 1:
        with tf.device("/cpu:0"):
            # convert the model to a model that can be trained with N_GPU GPUs
            model = keras.utils.multi_gpu_model(model, gpus=N_gpu)

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model
