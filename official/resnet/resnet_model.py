# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from enum import Enum

import numpy as np
import tensorflow as tf


class ConvType(Enum):
    STANDARD = 1
    SPECTRAL_PARAM = 2


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES
DEFAULT_CONV_TYPE = ConvType.SPECTRAL_PARAM
DEFAULT_RANDOM_SEED = 31


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        center=True,
        scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      data_format: The input format ('channels_last' or 'channels_first').

    Returns:
      A tensor with the same format as the input with the data either intact
      (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


class Model(object):
    """Base class for building the Resnet Model."""

    def __init__(self, resnet_size, bottleneck, num_classes, num_filters,
                 kernel_size, conv_stride, first_pool_size, first_pool_stride,
                 block_sizes, block_strides, final_size,
                 resnet_version=DEFAULT_VERSION, data_format=None,
                 dtype=DEFAULT_DTYPE, conv_type=DEFAULT_CONV_TYPE,
                 random_seed=DEFAULT_RANDOM_SEED):
        """Creates a model for classifying an image.

        Args:
          resnet_size: A single integer for the size of the ResNet model.
          bottleneck: Use regular blocks or bottleneck blocks.
          num_classes: The number of classes used as labels.
          num_filters: The number of filters to use for the first block layer
            of the model. This number is then doubled for each subsequent block
            layer.
          kernel_size: The kernel size to use for convolution.
          conv_stride: stride size for the initial convolutional layer
          first_pool_size: Pool size to be used for the first pooling layer.
            If none, the first pooling layer is skipped.
          first_pool_stride: stride size for the first pooling layer. Not used
            if first_pool_size is None.
          block_sizes: A list containing n values, where n is the number of
            sets of block layers desired. Each value should be the number of
            blocks in the i-th set.
          block_strides: List of integers representing the desired stride size
            for each of the sets of block layers. Should be same length as
            block_sizes.
          final_size: The expected size of the model after the second pooling.
          resnet_version: Integer representing which version of the ResNet
            network to use. See README for details. Valid values: [1, 2]
          data_format: Input format ('channels_last', 'channels_first',
            or None). If set to None, the format is dependent on whether a GPU
            is available.
          dtype: The TensorFlow dtype to use for calculations. If not specified
            tf.float32 is used.
          conv_type: The type of the convolution (for example, a standard one
            and the one with parameter initialization in the spectral domain).

        Raises:
          ValueError: if invalid version is selected.
        """
        self.resnet_size = resnet_size

        if not data_format:
            # data_format = (
            #     'channels_first' if tf.test.is_built_with_cuda() else
            #     'channels_last')
            data_format = "channels_last"  # this is the default format in tf

        self.resnet_version = resnet_version
        if resnet_version not in (1, 2):
            raise ValueError(
                'Resnet version should be 1 or 2. See README for citations.')

        self.bottleneck = bottleneck
        if bottleneck:
            if resnet_version == 1:
                self.block_fn = self._bottleneck_block_v1
            else:
                self.block_fn = self._bottleneck_block_v2
        else:
            if resnet_version == 1:
                self.block_fn = self._building_block_v1
            else:
                self.block_fn = self._building_block_v2

        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.final_size = final_size
        self.dtype = dtype
        self.pre_activation = resnet_version == 2
        self.conv_type = conv_type
        self.random_seed = random_seed

    def _custom_dtype_getter(self, getter, name, shape=None,
                             dtype=DEFAULT_DTYPE,
                             *args, **kwargs):
        """Creates variables in fp32, then casts to fp16 if necessary.

        This function is a custom getter. A custom getter is a function with the
        same signature as tf.get_variable, except it has an additional getter
        parameter. Custom getters can be passed as the `custom_getter` parameter
        of
        tf.variable_scope. Then, tf.get_variable will call the custom getter,
        instead of directly getting a variable itself. This can be used to
        change the types of variables that are retrieved with tf.get_variable.
        The `getter` parameter is the underlying variable getter, that would
        have been called if no custom getter was used. Custom getters typically
        get a variable with `getter`, then modify it in some way.

        This custom getter will create an fp32 variable. If a low precision
        (e.g. float16) variable was requested it will then cast the variable to
        the
        requested dtype. The reason we do not directly create variables in low
        precision dtypes is that applying small gradients to such variables may
        cause the variable not to change.

        Args:
          getter: The underlying variable getter, that has the same signature as
            tf.get_variable and returns a variable.
          name: The name of the variable to get.
          shape: The shape of the variable to get.
          dtype: The dtype of the variable to get. Note that if this is a low
            precision dtype, the variable will be created as a tf.float32
            variable, then cast to the appropriate dtype
          *args: Additional arguments to pass unmodified to getter.
          **kwargs: Additional keyword arguments to pass unmodified to getter.

        Returns:
          A variable which is cast to fp16 if necessary.
        """

        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self, name):
        """Returns a variable scope that the model should be created under.

        If self.dtype is a castable type, model variable will be created in fp32
        then cast to self.dtype before being used.

        Args:
          name: The name of the variable scope.

        Returns:
          A variable scope for the model.
        """

        return tf.variable_scope(name, custom_getter=self._custom_dtype_getter)

    def __call__(self, inputs, training):
        """Add operations to classify a batch of input images.

        Args:
          inputs: A Tensor representing a batch of input images.
          training: A boolean. Set to True to add operations required only when
            training the classifier.

        Returns:
          A logits Tensor with shape [<batch_size>, self.num_classes].
        """

        with self._model_variable_scope('resnet_model'):
            # if self.data_format == 'channels_first':
            #     # Convert the inputs from channels_last (NHWC) to
            #     # channels_first (NCHW).
            #     # This provides a large performance boost on GPU. See
            #     # https://www.tensorflow.org/performance/performance_guide#
            #     # data_formats
            #     inputs = tf.transpose(inputs, [0, 3, 1, 2])

            with self._model_variable_scope('initial_conv2d'):
                inputs = self.conv2d_fixed_padding(
                    inputs=inputs, filters=self.num_filters,
                    kernel_size=self.kernel_size,
                    strides=self.conv_stride)
            inputs = tf.identity(inputs, 'initial_identity')

            # We do not include batch normalization or activation functions in
            # V2 for the initial conv1 because the first ResNet unit will
            # perform these for both the shortcut and non-shortcut paths as
            # part of the first block's projection. Cf. Appendix of [2].
            if self.resnet_version == 1:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)

            if self.first_pool_size:
                inputs = tf.layers.max_pooling2d(
                    inputs=inputs, pool_size=self.first_pool_size,
                    strides=self.first_pool_stride, padding='SAME',
                    data_format=self.data_format)
                inputs = tf.identity(inputs, 'initial_max_pool')

            for i, num_blocks in enumerate(self.block_sizes):
                num_filters = self.num_filters * (2 ** i)
                with self._model_variable_scope('block_layer_{}'.format(i + 1)):
                    inputs = self.block_layer(
                        inputs=inputs, filters=num_filters,
                        bottleneck=self.bottleneck,
                        block_fn=self.block_fn, blocks=num_blocks,
                        strides=self.block_strides[i], training=training)

            # Only apply the BN and ReLU for model that does pre_activation in
            # each building/bottleneck block, eg resnet V2.
            if self.pre_activation:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)

            # The current top layer has shape
            # `batch_size x pool_size x pool_size x final_size`.
            # ResNet does an Average Pooling layer over pool_size,
            # but that is the same as doing a reduce_mean. We do a reduce_mean
            # here because it performs better than AveragePooling2D.
            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(inputs, axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')

            inputs = tf.reshape(inputs, [-1, self.final_size])
            inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
            inputs = tf.identity(inputs, 'final_dense')
            return inputs

    def _glorot_sample(self, kernel_size, in_channel, filters):
        """
        The same definition of the glorot initialization as in tensorflow but
        for not a variable but for a separate sample.

        :param in_channel: Number of input channels for an image (typically 3
          for RGB or 1 for a gray scale image).
        :param filters: Number of filters in a layer
        :return: numpy array with glorot initialized values
        """
        limit = np.sqrt(6 / (in_channel + filters))
        return np.random.uniform(
            low=-limit, high=limit,
            size=(in_channel, filters, kernel_size, kernel_size))

    def conv2d_spectral_param(self, inputs, filters, kernel_size, strides,
                              padding, use_bias=False):
        """
        A convolutional layer with spectrally-parameterized weights.

        :param inputs: Should be a 4D array like:
                            (batch_num, channel_num, img_len, img_len)
        :param in_channel: The number of channels
        :param filters: Number of filters required
        :param kernel_size: Kernel size
        :param random_seed: Random seed
        :param data_format: Image should be with CHANNEL LAST: NHWC
        :param index: The layer index used for naming
        """
        if self.data_format == 'channels_last':
            in_channel = inputs.shape[3].value
            strides = [1, strides, strides, 1]
            data_format = "NHWC"
        elif self.data_format == 'channels_first':
            in_channel = inputs.shape[1].value
            strides = [1, 1, strides, strides]
            data_format = "NCHW"

        with self._model_variable_scope('spec_conv_layer'):
            with self._model_variable_scope('spec_conv_kernel'):
                glorot_sample = self._glorot_sample(kernel_size, in_channel,
                                                    filters)
                """
                tf.fft2d: Computes the 2-dimensional discrete Fourier transform 
                over the inner-most 2 dimensions of input.
                """
                # shape channel_in, channel_out, kernel_size, kernel_size
                spectral_weight_init = tf.fft2d(glorot_sample)

                real_init = tf.get_variable(
                    name='real', initializer=tf.real(spectral_weight_init))

                imag_init = tf.get_variable(
                    name='imag', initializer=tf.imag(spectral_weight_init))

                spectral_weight = tf.complex(
                    real_init, imag_init, name='spectral_weight')

            """
            ifft2d: Computes the inverse 2-dimensional discrete Fourier 
            transform over the inner-most 2 dimensions of input.
            """
            complex_spatial_weight = tf.ifft2d(spectral_weight)
            spatial_weight = tf.real(complex_spatial_weight,
                                     name='spatial_weight')

            # we need kernel tensor of shape [filter_height, filter_width,
            # in_channels, out_channels]
            spatial_weight = tf.transpose(spatial_weight, [2, 3, 0, 1])

            out = tf.nn.conv2d(inputs, spatial_weight, strides=strides,
                               padding=padding, data_format=data_format)

            if use_bias:
                with self._model_variable_scope('spec_conv_bias'):
                    b_shape = [filters]
                    bias = tf.get_variable(
                        name='spec_conv_bias',
                        shape=b_shape,
                        initializer=tf.glorot_uniform_initializer(
                            seed=self.random_seed
                        ))
                    out = tf.nn.bias_add(out, bias, data_format=data_format)

            return out

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides):
        """Strided 2-D convolution with explicit padding.

        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          filters: The number of filters for the convolutions.
          kernel_size: The kernel size to use for convolution (e.g. kernel
            size = 3 induces a kernel of spatial shape: 3 x 3).
          strides: The block's stride. If greater than 1, this block will
            ultimately downsample the input.

        Returns:
          The output tensor of the convolution.
        """
        # The padding is consistent and is based only on `kernel_size`, not on
        # the dimensions of `inputs` (as opposed to using `tf.layers.conv2d`
        # alone).
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size, self.data_format)

        if self.conv_type is ConvType.STANDARD:
            tf.logging.info("conv_type: " + self.conv_type.name)
            return tf.layers.conv2d(
                inputs=inputs, filters=filters, kernel_size=kernel_size,
                strides=strides,
                padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
                kernel_initializer=tf.variance_scaling_initializer(),
                data_format=self.data_format)
        elif self.conv_type is ConvType.SPECTRAL_PARAM:
            tf.logging.info("conv_type: " + self.conv_type.name)
            return self.conv2d_spectral_param(
                inputs=inputs, filters=filters, kernel_size=kernel_size,
                strides=strides, padding=('SAME' if strides == 1 else 'VALID'),
                use_bias=True)
        else:
            raise ValueError('conv_format should be: ' + ",".join(
                [conv_type.name for conv_type in ConvType]))

    ############################################################################
    # ResNet block definitions.
    ############################################################################
    def _building_block_v1(self, inputs, filters, training, projection_shortcut,
                           strides):
        """A single block for ResNet v1, without a bottleneck.

        Convolution then batch normalization then ReLU as described by:
          Deep Residual Learning for Image Recognition
          https://arxiv.org/pdf/1512.03385.pdf
          by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          filters: The number of filters for the convolutions.
          training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
          projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
          strides: The block's stride. If greater than 1, this block will
            ultimately downsample the input.

        Returns:
          The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs

        if projection_shortcut is not None:
            with self._model_variable_scope("shortcut"):
                shortcut = projection_shortcut(inputs)
                shortcut = batch_norm(inputs=shortcut, training=training,
                                      data_format=self.data_format)

        with self._model_variable_scope("first_conv"):
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=strides)
            inputs = batch_norm(inputs, training, self.data_format)
            inputs = tf.nn.relu(inputs)

        with self._model_variable_scope("second_conv"):
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=1)
            inputs = batch_norm(inputs, training, self.data_format)

        inputs += shortcut
        inputs = tf.nn.relu(inputs)

        return inputs

    def _building_block_v2(self, inputs, filters, training, projection_shortcut,
                           strides):
        """A single block for ResNet v2, without a bottleneck.

        Batch normalization then ReLu then convolution as described by:
          Identity Mappings in Deep Residual Networks
          https://arxiv.org/pdf/1603.05027.pdf
          by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          filters: The number of filters for the convolutions.
          training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
          projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
          strides: The block's stride. If greater than 1, this block will
            ultimately downsample the input.

        Returns:
          The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = tf.nn.relu(inputs)

        # The projection shortcut should come after the first batch norm and
        # ReLU since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            with self._model_variable_scope("shortcut"):
                shortcut = projection_shortcut(inputs)

        with self._model_variable_scope("first_conv"):
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=strides)
            inputs = batch_norm(inputs, training, self.data_format)
            inputs = tf.nn.relu(inputs)

        with self._model_variable_scope("second_conv"):
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=1)

        return inputs + shortcut

    def _bottleneck_block_v1(self, inputs, filters, training,
                             projection_shortcut, strides):
        """A single block for ResNet v1, with a bottleneck.

        Similar to _building_block_v1(), except using the "bottleneck" blocks
        described in:
          Convolution then batch normalization then ReLU as described by:
            Deep Residual Learning for Image Recognition
            https://arxiv.org/pdf/1512.03385.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          filters: The number of filters for the convolutions.
          training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
          projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
          strides: The block's stride. If greater than 1, this block will
            ultimately downsample the input.

        Returns:
          The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs

        if projection_shortcut is not None:
            with self._model_variable_scope("shortcut"):
                shortcut = projection_shortcut(inputs)
                shortcut = batch_norm(inputs=shortcut, training=training)

        with self._model_variable_scope("first_conv"):
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=1, strides=1)
            inputs = batch_norm(inputs, training, data_format=self.data_format)
            inputs = tf.nn.relu(inputs)

        with self._model_variable_scope("second_conv"):
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=strides)
            inputs = batch_norm(inputs, training, data_format=self.data_format)
            inputs = tf.nn.relu(inputs)

        with self._model_variable_scope("third_conv"):
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)
            inputs = batch_norm(inputs, training, data_format=self.data_format)
            inputs += shortcut
            inputs = tf.nn.relu(inputs)

        return inputs

    def _bottleneck_block_v2(self, inputs, filters, training,
                             projection_shortcut, strides):
        """A single block for ResNet v2, without a bottleneck.

        Similar to _building_block_v2(), except using the "bottleneck" blocks
        described in:
          Convolution then batch normalization then ReLU as described by:
            Deep Residual Learning for Image Recognition
            https://arxiv.org/pdf/1512.03385.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

        Adapted to the ordering conventions of:
          Batch normalization then ReLu then convolution as described by:
            Identity Mappings in Deep Residual Networks
            https://arxiv.org/pdf/1603.05027.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          filters: The number of filters for the convolutions.
          training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
          projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
          strides: The block's stride. If greater than 1, this block will
            ultimately downsample the input.

        Returns:
          The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs
        inputs = batch_norm(inputs, training, data_format=self.data_format)
        inputs = tf.nn.relu(inputs)

        # The projection shortcut should come after the first batch norm and
        # ReLU since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            with self._model_variable_scope("shortcut"):
                shortcut = projection_shortcut(inputs)

        with self._model_variable_scope("first_conv"):
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=1, strides=1)
            inputs = batch_norm(inputs, training, data_format=self.data_format)
            inputs = tf.nn.relu(inputs)

        with self._model_variable_scope("second_conv"):
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=filters, kernel_size=3, strides=strides)
            inputs = batch_norm(inputs, training, data_format=self.data_format)
            inputs = tf.nn.relu(inputs)

        with self._model_variable_scope("third_conv"):
            inputs = self.conv2d_fixed_padding(
                inputs=inputs, filters=4 * filters, kernel_size=1, strides=1)

        return inputs + shortcut

    def block_layer(self, inputs, filters, bottleneck, block_fn, blocks,
                    strides, training):
        """Creates one layer of blocks for the ResNet model.

        Args:
          inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
          filters: The number of filters for the first convolution of the layer.
          bottleneck: Is the block created a bottleneck block.
          block_fn: The block to use within the model, either `building_block`
            or `bottleneck_block`.
          blocks: The number of blocks contained in the layer.
          strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
          training: Either True or False, whether we are currently training the
            model. Needed for batch norm.

        Returns:
          The output tensor of the block layer.
        """

        # Bottleneck blocks end with 4x the number of filters as they start with
        filters_out = filters * 4 if bottleneck else filters

        def projection_shortcut(inputs):
            return self.conv2d_fixed_padding(inputs=inputs, filters=filters_out,
                                             kernel_size=1, strides=strides)

        # Only the first block per block_layer uses projection_shortcut and
        # strides.
        with self._model_variable_scope('projection_shortcut'):
            inputs = block_fn(inputs, filters, training, projection_shortcut,
                              strides)

        for index in range(1, blocks):
            with self._model_variable_scope('block_{}'.format(index)):
                inputs = block_fn(inputs, filters, training, None, 1)

        return tf.identity(inputs, 'identity')
