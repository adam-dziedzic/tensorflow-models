# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
# pylint: disable=unused-import
"""DenseNet models for Keras.

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.applications.imagenet_utils import \
    _obtain_input_shape
from tensorflow.python.keras.engine.network import get_source_inputs
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.data_utils import get_file

from .utils import ConvType, get_conv_2D, EnumWithNames

DENSENET121_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET121_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
DENSENET169_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet169_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET169_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
DENSENET201_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels.h5'
DENSENET201_WEIGHT_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'

DEFAULT_CONV_TYPE = ConvType.STANDARD


class DenseNetSize(EnumWithNames):
    DENSE_NET_121 = 1
    DENSE_NET_169 = 2
    DENSE_NET_201 = 3


def conv_2D(inputs, filters, kernel_size, name, conv_type, padding='valid',
            strides=1, use_bias=False, data_format="channels_last"):
    """
    Get the 2D convolutional layer.

    :param inputs: the input layers (maps) to the convolution.
    :param filters: the number of filters for the layer.
    :param kernel_size: the size of the kernel.
    :param name: the name of the layer
    :param strides: the stride of the convolution, how many pixels should we
    move through for each filter
    :param use_bias: if bias should be added to the output of a convolution
    with a filter
    :return: the 2D convolutional layer
    """
    if conv_type == ConvType.STANDARD:
        return Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      padding=padding, use_bias=use_bias, name=name,
                      data_format=data_format)(inputs)
    else:
        return Lambda(get_conv_2D)(
            inputs=inputs, filters=filters, kernel_size=kernel_size,
            strides=strides, padding=padding, use_bias=use_bias, name=name,
            conv_type=conv_type, data_format=data_format)


def dense_block(x, blocks, name, conv_type, data_format="channels_last"):
    """A dense block.

    Arguments:
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
        conv_type: the type of parameters for convolution - initialized either
        in spectral or spatial domain.
        data_format: channels first or last.

    Returns:
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=name + '_block' + str(i + 1),
                       conv_type=conv_type, data_format=data_format)
    return x


def transition_block(x, reduction, name, conv_type,
                     data_format="channels_last"):
    """A transition block.

    Arguments:
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
        data_format: channels last or first

    Returns:
        output tensor for the block.
    """
    # batch normalization axes
    bn_axis = 3 if data_format == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = conv_2D(
        inputs=x, filters=int(K.int_shape(x)[bn_axis] * reduction),
        kernel_size=1, use_bias=False, name=name + '_conv',
        data_format=data_format, conv_type=conv_type)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x


class ConvSpectralSpatial2D(object):
    """
    Convolutional layer with spectral and spatial parameter initialization.
    """

    def __init__(self, filters, kernel_size, use_bias, name):
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.name = name

    def __call__(self, inputs):
        return inputs


def conv_block(x, growth_rate, name, conv_type, data_format="channels_last"):
    """A building block for a dense block.

    Arguments:
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
        conv_type: the type of initialization of weights for the convolution,
        either in spectral or in spatial domain.
        data_format: channels first or last

    Returns:
        output tensor for the block.
    """
    bn_axis = 3 if data_format == 'channels_last' else 1
    x1 = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
        x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = conv_2D(inputs=x1, filters=4 * growth_rate, kernel_size=1,
                 use_bias=False, name=name + '_1_conv', data_format=data_format,
                 conv_type=conv_type)
    x1 = BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(
        x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = conv_2D(inputs=x1, filters=growth_rate, kernel_size=3, padding='same',
                 use_bias=False, name=name + '_2_conv', data_format=data_format,
                 conv_type=conv_type)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(blocks,
             conv_type,
             include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=10,
             default_size=32,  # should be 224 for imagenet
             min_size=32,  # should be 221 for imagenet
             data_format="channels_last"):
    """Instantiates the DenseNet architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with
    TensorFlow, Theano, and CNTK. The data format
    convention used by the model is the one
    specified in your Keras config file.

    Arguments:
        blocks: numbers of building blocks for the four dense layers.
        conv_type: initialize spectral or spatial parameters.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        default_size: Default input width/height for the model.
        min_size: Minimum input width/height accepted by the model.
        data_format: Should the channels be provided as the last of first
        dimension of a given image.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    tf.logging.info("conv_type in dense_net: " + conv_type.name)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=min_size,
        data_format=data_format,
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if data_format == 'channels_last' else 1

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = conv_2D(inputs=x, filters=64, kernel_size=7, strides=2, use_bias=False,
                name='conv1/conv', data_format=data_format, conv_type=conv_type)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2', data_format=data_format,
                    conv_type=conv_type)
    x = transition_block(x, 0.5, name='pool2', data_format=data_format,
                         conv_type=conv_type)
    x = dense_block(x, blocks[1], name='conv3', data_format=data_format,
                    conv_type=conv_type)
    x = transition_block(x, 0.5, name='pool3', data_format=data_format,
                         conv_type=conv_type)
    x = dense_block(x, blocks[2], name='conv4', data_format=data_format,
                    conv_type=conv_type)
    x = transition_block(x, 0.5, name='pool4', data_format=data_format,
                         conv_type=conv_type)
    x = dense_block(x, blocks[3], name='conv5', data_format=data_format,
                    conv_type=conv_type)

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc' + str(classes))(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='densenet201')
    else:
        model = Model(inputs, x, name='densenet')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if blocks == [6, 12, 24, 16]:
                weights_path = get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET121_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='0962ca643bae20f9b6771cb844dca3b0')
            elif blocks == [6, 12, 32, 32]:
                weights_path = get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET169_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='bcf9965cf5064a5f9eb6d7dc69386f43')
            elif blocks == [6, 12, 48, 32]:
                weights_path = get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels.h5',
                    DENSENET201_WEIGHT_PATH,
                    cache_subdir='models',
                    file_hash='7bb75edd58cb43163be7e0005fbe95ef')
        else:
            if blocks == [6, 12, 24, 16]:
                weights_path = get_file(
                    'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET121_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='4912a53fbd2a69346e7f2c0b5ec8c6d3')
            elif blocks == [6, 12, 32, 32]:
                weights_path = get_file(
                    'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET169_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='50662582284e4cf834ce40ab4dfa58c6')
            elif blocks == [6, 12, 48, 32]:
                weights_path = get_file(
                    'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    DENSENET201_WEIGHT_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='1c2de60ee40562448dbac34a0737e798')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def DenseNet121(conv_type=ConvType.STANDARD,
                include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=10,
                default_size=32,
                min_size=32):
    tf.logging.info("Chosen DenseNet121")
    return DenseNet(blocks=[6, 12, 24, 16], conv_type=conv_type,
                    include_top=include_top, weights=weights,
                    input_tensor=input_tensor, input_shape=input_shape,
                    pooling=pooling, classes=classes, default_size=default_size,
                    min_size=min_size)


def DenseNet169(conv_type=ConvType.STANDARD,
                include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=10,
                default_size=32,
                min_size=32):
    tf.logging.info("Chosen DenseNet169")
    return DenseNet(blocks=[6, 12, 32, 32], conv_type=conv_type,
                    include_top=include_top, weights=weights,
                    input_tensor=input_tensor, input_shape=input_shape,
                    pooling=pooling, classes=classes, default_size=default_size,
                    min_size=min_size)


def DenseNet201(conv_type=ConvType.STANDARD,
                include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=10,
                default_size=32,
                min_size=32):
    tf.logging.info("Chosen DenseNet201")
    return DenseNet(blocks=[6, 12, 48, 32], conv_type=conv_type,
                    include_top=include_top, weights=weights,
                    input_tensor=input_tensor, input_shape=input_shape,
                    pooling=pooling, classes=classes, default_size=default_size,
                    min_size=min_size)


def preprocess_input(x, data_format=None):
    """Preprocesses a numpy array encoding a batch of images.

    Arguments:
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.

    Returns:
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format, mode='torch')


setattr(DenseNet121, '__doc__', DenseNet.__doc__)
setattr(DenseNet169, '__doc__', DenseNet.__doc__)
setattr(DenseNet201, '__doc__', DenseNet.__doc__)
