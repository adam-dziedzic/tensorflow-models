from enum import Enum

import numpy as np
import tensorflow as tf


class EnumWithNames(Enum):
    """
    The Enum classes that inherit from the EnumWithNames will get the get_names
    method to return an array of strings representing all possible enum values.
    """

    @classmethod
    def get_names(cls):
        return [enum_value.name for enum_value in cls]


class ConvType(EnumWithNames):
    STANDARD = 1
    SPECTRAL_PARAM = 2
    SPECTRAL_DIRECT = 3
    SPATIAL_PARAM = 4


def _glorot_sample(kernel_size, in_channel, filters):
    """
    The same definition of the glorot initialization as in tensorflow but
    for not a variable but for a separate sample.

    :param kernel_size: The width and length of the filter (kernel).
    :param in_channel: Number of input channels for an image (typically 3
      for RGB or 1 for a gray scale image).
    :param filters: Number of filters in a layer
    :return: numpy array with glorot initialized values
    """
    limit = np.sqrt(6 / (in_channel + filters))
    return np.random.uniform(
        low=-limit, high=limit,
        size=(in_channel, filters, kernel_size, kernel_size))


def get_spatial_weights(kernel_size, in_channel, filters, conv_type, dtype,
                        name=""):
    """
    Get the filter weights for the convolution, either spectral or spatial.

    :param kernel_size: the size of the kernel, with input value 3, the kernel
    is 3x3
    :param in_channel: the number of channels in the input to the convolution
    (e.g. the initial
    image with RGB - 3 channels).
    :param filters: the number of filters in the convolution
    :param conv_type: the type of weight initialization for the convolution:
    spectral or spatial
    :param dtype: the type of the input (e.g. float32 or float64).
    :param name: the name prefix for the variables created in the method
    :return: the initialized weights in the graph
    """
    tf.logging.info("name value in utils: " + name)
    glorot_sample = _glorot_sample(kernel_size, in_channel, filters)
    if conv_type == ConvType.SPATIAL_PARAM:
        glorot_sample = tf.convert_to_tensor(glorot_sample, dtype)
        # we need kernel tensor of shape [filter_height, filter_
        # width, in_channels, out_channels] Here we do it only once,
        # for initialization.
        glorot_sample = tf.transpose(glorot_sample, [2, 3, 0, 1])
        spatial_weight = tf.get_variable(
            name=name + '_spatial_weight', initializer=glorot_sample)
    else:
        """
        tf.fft2d: Computes the 2-dimensional discrete Fourier 
        transform over the inner-most 2 dimensions of input.
        """
        tf.logging.info("utils nets initialize spectral weights")
        # shape channel_in, channel_out, kernel_size, kernel_size
        spectral_weight_init = tf.fft2d(glorot_sample)

        if conv_type is ConvType.SPECTRAL_PARAM:

            real_init = tf.get_variable(
                name=name + 'real',
                initializer=tf.real(spectral_weight_init))

            imag_init = tf.get_variable(
                name=name + 'imag',
                initializer=tf.imag(spectral_weight_init))

            spectral_weight = tf.complex(
                real_init, imag_init, name=name + 'spectral_weight')

        elif conv_type is ConvType.SPECTRAL_DIRECT:
            tf.logging.ERROR("This does not work, the type of "
                             "paramaters should be in [tf.float32, "
                             "tf.float64, tf.float16, tf.bfloat16]")
            spectral_weight = tf.get_variable(
                name= name + 'spectral_param',
                initializer=spectral_weight_init)

        else:
            raise ValueError('conv_type should be: ' + ",".join(
                [conv_type.name for conv_type in ConvType]))

        """
        ifft2d: Computes the inverse 2-dimensional discrete Fourier 
        transform over the inner-most 2 dimensions of input.
        """
        complex_spatial_weight = tf.ifft2d(spectral_weight)
        spatial_weight = tf.real(complex_spatial_weight,
                                 name=name + 'spatial_weight')

        # we need kernel tensor of shape [filter_height, filter_
        # width, in_channels, out_channels]
        spatial_weight = tf.transpose(spatial_weight, [2, 3, 0, 1])

    return spatial_weight


def get_conv_2D(inputs, kernel_size, filters, conv_type, padding, name="",
                strides=1, use_bias=False, data_format="NHWC", random_seed=31):
    """
    Get the convolutional layer.

    :param inputs:
    :param kernel_size: the size of the kernel, with input value 3, the kernel
    is 3x3
    :param filters: the number of filters in the convolution
    :param conv_type: the type of weight initialization for the convolution:
    spectral or spatial
    :param strides: the stride value for the convolution
    :param conv_type: the convolution with spectral or spatial param
    initiliaztion
    :param conv_type: type of parameters for the convolution, either spectral
    or spatial
    :param name: the prefix name for the variables
    :param use_bias: should we add the bias to the convolution or not
    :param random_seed: the initial number to initialize random generator
    :param data_format: where to put the channel, as last of second dimension
    :return: the convolutional layer
    """

    if data_format == 'channels_last':
        in_channel = inputs.shape[3].value
        strides = [1, strides, strides, 1]
        data_format = "NHWC"
    elif data_format == 'channels_first':
        in_channel = inputs.shape[1].value
        strides = [1, 1, strides, strides]
        data_format = "NCHW"

    spatial_weights = get_spatial_weights(
        kernel_size=kernel_size, in_channel=in_channel,
        filters=filters, conv_type=conv_type,
        dtype=inputs.dtype, name=name)

    out = tf.nn.conv2d(inputs, spatial_weights, strides=strides,
                       padding=padding.upper(), data_format=data_format,
                       name=name + "_spectral_spatial_conv2D")

    if use_bias:
        b_shape = [filters]
        bias = tf.get_variable(
            name=name + '_spectral_conv_bias',
            shape=b_shape,
            initializer=tf.glorot_uniform_initializer(
                seed=random_seed
            ))
        out = tf.nn.bias_add(out, bias, data_format=data_format)

    return out


def learning_rate_with_decay(
        batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
    """Get a learning rate that decays step-wise as training progresses.

    Args:
      batch_size: the number of examples processed in each training batch.
      batch_denom: this value will be used to scale the base learning rate.
        `0.1 * batch size` is divided by this number, such that when
        batch_denom == batch_size, the initial learning rate will be 0.1.
      num_images: total number of images that will be used for training.
      boundary_epochs: list of ints representing the epochs at which we
        decay the learning rate.
      decay_rates: list of floats representing the decay rates to be used
        for scaling the learning rate. It should have one more element
        than `boundary_epochs`, and all elements should have the same type.

    Returns:
      Returns a function that takes a single argument - the number of batches
      trained so far (global_step)- and returns the learning rate to be used
      for training the next batch.
    """
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Reduce the learning rate at certain epochs.
    # CIFAR-10: divide by 10 at epoch 100, 150, and 200
    # ImageNet: divide by 10 at epoch 30, 60, 80, and 90
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn
