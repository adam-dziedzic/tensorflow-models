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
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
# pylint: disable=g-bad-import-order
from absl import flags

from official.nets import densenet_model
from official.nets import resnet_model
from official.utils.export import export
from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers
from .utils import EnumWithNames
from .utils import BatchNorm


# pylint: enable=g-bad-import-order


class OptimizerType(EnumWithNames):
    MOMENTUM = 1
    ADAM = 2


class RunType(EnumWithNames):
    TEST = 0
    DEBUG = 1


class ModelType(EnumWithNames):
    RES_NET = 0
    DENSE_NET = 1


DEFAULT_OPTIMIZER = OptimizerType.ADAM
DEFAULT_RUN_TYPE = RunType.TEST
DEFAULT_MODEL_TYPE = ModelType.RES_NET
DEFAULT_DATA_FORMAT = "channels_last"


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None,
                           data_format=DEFAULT_DATA_FORMAT):
    """Given a Dataset with raw records, return an iterator over the records.

    Args:
      dataset: A Dataset representing raw records
      is_training: A boolean denoting whether the input is for training.
      batch_size: The number of samples per batch.
      shuffle_buffer: The buffer size to use when shuffling records. A larger
        value results in better randomness, but smaller values reduce startup
        time and use less memory.
      parse_record_fn: A function that takes a raw record and returns the
        corresponding (image, label) pair.
      num_epochs: The number of epochs to repeat the dataset.
      num_gpus: The number of gpus used for training.
      examples_per_epoch: The number of examples in an epoch.
      data_format: The format of the data: 'channels_last' or 'channels_first'.

    Returns:
      Dataset of (image, label) pairs ready for iteration.
    """

    # We prefetch a batch at a time, This can help smooth out the time taken to
    # load input files as we go through shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffle the records. Note that we shuffle before repeating to ensure
        # that the shuffling respects epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # If we are training over multiple epochs before evaluating, repeat the
    # dataset for the appropriate number of epochs.
    dataset = dataset.repeat(num_epochs)

    if is_training and num_gpus and examples_per_epoch:
        total_examples = num_epochs * examples_per_epoch
        # Force the number of batches to be divisible by the number of devices.
        # This prevents some devices from receiving batches while others do not,
        # which can lead to a lockup. This case will soon be handled directly by
        # distribution strategies, at which point this .take() operation will no
        # longer be needed.
        total_batches = total_examples // batch_size // num_gpus * num_gpus
        dataset.take(total_batches * batch_size)

    # Parse the raw records into images and labels. Testing has shown that setting
    # num_parallel_batches > 1 produces no improvement in throughput, since
    # batch_size is almost always much greater than the number of CPU cores.
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: parse_record_fn(value, is_training, data_format),
            batch_size=batch_size,
            num_parallel_batches=1,
            drop_remainder=False))

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
    # allows DistributionStrategies to adjust how many batches to fetch based
    # on how many devices are present.
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset


def get_synth_input_fn(height, width, num_channels, num_classes,
                       data_format=DEFAULT_DATA_FORMAT):
    # pylint: disable=unused-argument
    """Returns an input function that returns a dataset with zeroes.

    This is useful in debugging input pipeline performance, as it removes all
    elements of file reading and image preprocessing.

    Args:
      height: Integer height that will be used to create a fake image tensor.
      width: Integer width that will be used to create a fake image tensor.
      num_channels: Integer depth that will be used to create a fake image tensor.
      num_classes: Number of classes that should be represented in the fake labels
        tensor
      data_format: The data format type, either 'channels_last' or
        'channels_first'

    Returns:
      An input_fn that can be used in place of a real one to return a dataset
      that can be used for iteration.
    """

    def input_fn(is_training, data_dir, batch_size, data_format=data_format,
                 *args, **kwargs):  # pylint: disable=unused-argument
        if data_format == "channels_first":
            input_shape = tf.TensorShape(
                [batch_size, num_channels, height, width])
        else:  # channels_last
            input_shape = tf.TensorShape(
                [batch_size, height, width, num_channels])
        return model_helpers.generate_synthetic_data(
            input_shape=input_shape,
            input_dtype=tf.float32,
            label_shape=tf.TensorShape([batch_size]),
            label_dtype=tf.int32)

    return input_fn


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def net_model_fn(features, labels, mode, model,
                 resnet_size, weight_decay, learning_rate_fn, momentum,
                 data_format, resnet_version, loss_scale,
                 loss_filter_fn=None, dtype=resnet_model.DEFAULT_DTYPE,
                 conv_type=resnet_model.DEFAULT_CONV_TYPE,
                 optimizer_type=DEFAULT_OPTIMIZER,
                 run_type=DEFAULT_RUN_TYPE):
    """Shared functionality for different nets model_fns.

    Uses that model to build the necessary EstimatorSpecs for
    the `mode` in question. For training, this means building losses,
    the optimizer, and the train op that get passed into the EstimatorSpec.
    For evaluation and prediction, the EstimatorSpec is returned without
    a train op, but with the necessary parameters for the given mode.

    Args:
      features: tensor representing input images
      labels: tensor representing class labels for all input images
      mode: current estimator mode; should be one of
        `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
      model: a TensorFlow model that has a __call__ function.
      resnet_size: A single integer for the size of the ResNet model.
      weight_decay: weight decay loss rate used to regularize learned variables.
      learning_rate_fn: function that returns the current learning rate given
        the current global_step
      momentum: momentum term used for optimization
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      resnet_version: Integer representing which version of the ResNet network to
        use. See README for details. Valid values: [1, 2]
      loss_scale: The factor to scale the loss for numerical stability.
        A detailed summary is present in the arg parser help text.
      loss_filter_fn: function that takes a string variable name and returns
        True if the var should be included in loss calculation, and False
        otherwise. If None, batch_normalization variables will be excluded
        from the loss.
      dtype: the TensorFlow dtype to use for calculations.

    Returns:
      EstimatorSpec parameterized according to the input params and the
      current mode.
    """

    # Generate a summary node for the images
    # `tensor` features which must be 4-D with shape `[batch_size, height,
    # width, channels]`
    tf.summary.image('images', features, max_outputs=6)

    features = tf.cast(features, dtype)

    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    # This acts as a no-op if the logits are already in fp32 (provided logits are
    # not a SparseTensor). If dtype is is low precision, logits must be cast to
    # fp32 for numerical stability.
    logits = tf.cast(logits, tf.float32)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    loss_filter_fn = loss_filter_fn or exclude_batch_norm

    # Add weight decay to the loss.
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        tf.logging.info("optimizer_type: " + optimizer_type.name)
        if optimizer_type is OptimizerType.MOMENTUM:
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=momentum
            )
        elif optimizer_type is OptimizerType.ADAM:
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                epsilon=1e-8
            )
        else:
            raise ValueError(
                "Unsupported optimizer type: " + str(optimizer_type) +
                ". Please choose from: " + ",".join(
                    [optimizer_type.name for optimizer_type in OptimizerType]))

        if loss_scale != 1:
            # When computing fp16 gradients, often intermediate tensor values are
            # so small, they underflow to 0. To avoid this, we multiply the loss by
            # loss_scale to make these tensor values loss_scale times bigger.
            scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

            # Once the gradient computation is complete we can scale the gradients
            # back to the correct scale before passing them to the optimizer.
            unscaled_grad_vars = [(grad / loss_scale, var)
                                  for grad, var in scaled_grad_vars]
            minimize_op = optimizer.apply_gradients(unscaled_grad_vars,
                                                    global_step)
        else:
            minimize_op = optimizer.minimize(loss, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    if not tf.contrib.distribute.has_distribution_strategy():
        accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    else:
        # Metrics are currently not compatible with distribution strategies during
        # training. This does not affect the overall performance of the model.
        accuracy = (tf.no_op(), tf.constant(0))

    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def net_main(
        flags_obj, model_function, input_function, dataset_name, shape=None):
    """Shared main loop for ResNet Models.

    Args:
      flags_obj: An object containing parsed flags. See define_resnet_flags()
        for details.
      model_function: the function that instantiates the Model and builds the
        ops for train/eval. This will be passed directly into the estimator.
      input_function: the function that processes the dataset and returns a
        dataset that the estimator can train on. This will be wrapped with
        all the relevant flags for running and passed to estimator.
      dataset_name: the name of the dataset for training and evaluation. This is
        used for logging purpose.
      shape: list of ints representing the shape of the images used for
        training.
        This is only used if flags_obj.export_dir is passed.
    """

    model_helpers.apply_clean(flags.FLAGS)

    # Using the Winograd non-fused algorithm provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Create session config based on values of inter_op_parallelism_threads and
    # intra_op_parallelism_threads. Note that we default to having
    # allow_soft_placement = True, which is required for multi-GPU and not
    # harmful for other modes.
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=flags_obj.inter_op_parallelism_threads,
        intra_op_parallelism_threads=flags_obj.intra_op_parallelism_threads,
        allow_soft_placement=True)

    distribution_strategy = distribution_utils.get_distribution_strategy(
        flags_core.get_num_gpus(flags_obj), flags_obj.all_reduce_alg)

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy, session_config=session_config)

    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir,
        config=run_config,
        params={
            'model_type': ModelType[flags_obj.model_type],
            'densenet_size': densenet_model.DenseNetSize[
                flags_obj.densenet_size],
            'resnet_size': int(flags_obj.resnet_size),
            'data_format': flags_obj.data_format,
            'batch_size': flags_obj.batch_size,
            'resnet_version': int(flags_obj.resnet_version),
            'loss_scale': flags_core.get_loss_scale(flags_obj),
            'dtype': flags_core.get_tf_dtype(flags_obj),
            'conv_type': resnet_model.ConvType[flags_obj.conv_type],
            'optimizer_type': OptimizerType[flags_obj.optimizer_type],
            'run_type': RunType[flags_obj.run_type],
            'batch_norm_state': BatchNorm[flags_obj.batch_norm_state]
        })

    run_params = {
        'model_type': ModelType[flags_obj.model_type],
        'densenet_size': densenet_model.DenseNetSize[
            flags_obj.densenet_size],
        'batch_size': flags_obj.batch_size,
        'dtype': flags_core.get_tf_dtype(flags_obj),
        'resnet_size': flags_obj.resnet_size,
        'resnet_version': flags_obj.resnet_version,
        'synthetic_data': flags_obj.use_synthetic_data,
        'train_epochs': flags_obj.train_epochs,
        'conv_type': resnet_model.ConvType[flags_obj.conv_type],
        'optimizer_type': OptimizerType[flags_obj.optimizer_type],
        'run_type': RunType[flags_obj.run_type],
        'batch_norm_state': BatchNorm[flags_obj.batch_norm_state]
    }
    if flags_obj.use_synthetic_data:
        dataset_name = dataset_name + '-synthetic'

    benchmark_logger = logger.get_benchmark_logger()
    benchmark_logger.log_run_info('nets', dataset_name, run_params,
                                  test_id=flags_obj.benchmark_test_id)

    train_hooks = hooks_helper.get_train_hooks(
        flags_obj.hooks,
        model_dir=flags_obj.model_dir,
        batch_size=flags_obj.batch_size)

    def input_fn_train():
        return input_function(
            is_training=True, data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
            num_epochs=flags_obj.epochs_between_evals,
            num_gpus=flags_core.get_num_gpus(flags_obj),
            data_format=flags_obj.data_format)

    def input_fn_eval_train():
        return input_function(
            is_training=True, data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
            num_epochs=1)

    def input_fn_eval_test():
        return input_function(
            is_training=False, data_dir=flags_obj.data_dir,
            batch_size=distribution_utils.per_device_batch_size(
                flags_obj.batch_size, flags_core.get_num_gpus(flags_obj)),
            num_epochs=1)

    total_training_cycle = (flags_obj.train_epochs //
                            flags_obj.epochs_between_evals)
    for cycle_index in range(total_training_cycle):
        tf.logging.info('Starting a training cycle: %d/%d',
                        cycle_index, total_training_cycle)

        classifier.train(input_fn=input_fn_train, hooks=train_hooks,
                         max_steps=flags_obj.max_train_steps)

        tf.logging.info('Starting to evaluate.')

        # flags_obj.max_train_steps is generally associated with testing and
        # profiling. As a result it is frequently called with synthetic data, which
        # will iterate forever. Passing steps=flags_obj.max_train_steps allows the
        # eval (which is generally unimportant in those circumstances) to terminate.
        # Note that eval will run for max_train_steps each loop, regardless of the
        # global_step count.
        eval_results = classifier.evaluate(input_fn=input_fn_eval_test,
                                           steps=flags_obj.max_train_steps)
        tf.logging.info('Test evaluation')
        benchmark_logger.log_evaluation_result(eval_results)

        eval_results = classifier.evaluate(input_fn=input_fn_eval_train,
                                           steps=flags_obj.max_train_steps)
        tf.logging.info('Train evaluation')
        benchmark_logger.log_evaluation_result(eval_results)

        if model_helpers.past_stop_threshold(
                flags_obj.stop_threshold, eval_results['accuracy']):
            break

    if flags_obj.export_dir is not None:
        # Exports a saved model for the given classifier.
        input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
            shape, batch_size=flags_obj.batch_size)
        classifier.export_savedmodel(flags_obj.export_dir, input_receiver_fn)


def define_nets_flags(resnet_size_choices=None):
    """Add flags and validators for Nets."""
    flags_core.define_base()
    flags_core.define_performance(num_parallel_calls=False)
    flags_core.define_image()
    flags_core.define_benchmark()
    flags.adopt_module_key_flags(flags_core)

    flags.DEFINE_enum(
        name='resnet_version', short_name='rv', default='2',
        enum_values=['1', '2'],
        help=flags_core.help_wrap(
            'Version of ResNet. (1 or 2) See README.md for details.'))

    conv_types = resnet_model.ConvType.get_names()
    flags.DEFINE_enum(
        name='conv_type', short_name='ct',
        # default='SPECTRAL_PARAM',
        default='STANDARD',
        enum_values=conv_types,
        help=flags_core.help_wrap(
            'Version of the convolution used. STANDARD is with default '
            'convolutional layer. SPECTRAL_PARAM initializes parameters in the'
            'frequency domain. SPATIAL_PARAM similar to the spectral version'
            ' but initializes parameters in the spatial domain.'))

    optimizer_types = OptimizerType.get_names()
    flags.DEFINE_enum(
        name='optimizer_type', short_name='opt', default='MOMENTUM',
        enum_values=optimizer_types,
        help=flags_core.help_wrap(
            'Version of the optimizer to use, e.g., ' + ",".join(
                optimizer_types)))

    model_types = ModelType.get_names()
    flags.DEFINE_enum(
        name='model_type', short_name='model', default='RES_NET',
        enum_values=model_types,
        help=flags_core.help_wrap(
            'Type of the model: ' + ",".join(model_types)))

    batch_norm_states = BatchNorm.get_names()
    flags.DEFINE_enum(
        name='batch_norm_state', short_name='bn', default='INACTIVE',
        enum_values=batch_norm_states,
        help=flags_core.help_wrap(
            'States of the batch norm: ' + ",".join(batch_norm_states)))

    densenet_sizes = densenet_model.DenseNetSize.get_names()
    flags.DEFINE_enum(
        name='densenet_size', short_name='densenet_size',
        default='DENSE_NET_121', enum_values=densenet_sizes,
        help=flags_core.help_wrap(
            'The size of the dense net model: ' + ",".join(densenet_sizes)))

    run_types = [run_type.name for run_type in RunType]
    flags.DEFINE_enum(
        name='run_type', short_name='run', default='TEST',
        enum_values=run_types,
        help=flags_core.help_wrap('Type of the run: ' + ",".join(run_types)))

    choice_kwargs = dict(
        name='resnet_size', short_name='rs', default='50',
        help=flags_core.help_wrap('The size of the ResNet model to use.'))

    if resnet_size_choices is None:
        flags.DEFINE_string(**choice_kwargs)
    else:
        flags.DEFINE_enum(enum_values=resnet_size_choices, **choice_kwargs)

    # The current implementation of ResNet v1 is numerically unstable when run
    # with fp16 and will produce NaN errors soon after training begins.
    msg = ('ResNet version 1 is not currently supported with fp16. '
           'Please use version 2 instead.')

    @flags.multi_flags_validator(['dtype', 'resnet_version'], message=msg)
    def _forbid_v1_fp16(flag_values):  # pylint: disable=unused-variable
        return (flags_core.DTYPE_MAP[flag_values['dtype']][0] != tf.float16 or
                flag_values['resnet_version'] != '1')
