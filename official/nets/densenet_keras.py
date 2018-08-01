import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from official.nets.densenet_model import DenseNet121, DenseNet169, DenseNet201
from official.nets.densenet_model import DenseNetSize
from official.nets.run_loop import OptimizerType
from official.nets.utils import ConvType, EnumWithNames, \
    learning_rate_with_decay


class ExecMode(EnumWithNames):
    """
    Execution mode.
    """
    DEBUG = 0
    TEST = 1


model_type = DenseNetSize.DENSE_NET_121
epochs = 300
mode = ExecMode.TEST
debug_limit_tuples = 256
batch_size = 64
verbosity = 0
conv_type = ConvType.SPECTRAL_PARAM
num_classes = 10
data_augmentation = False
optimizer_type = OptimizerType.ADAM
momentum_value = 0.9

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", default=epochs, type=int,
                    help="number of epochs for the training")
parser.add_argument("-t", "--debug_limit_tuples", default=debug_limit_tuples,
                    type=int, help="number of tuples used from the training "
                                   "and test datasets")
parser.add_argument("-c", "--conv_type", default=conv_type.name,
                    help="convolution type from: " + ",".join(
                        ConvType.get_names()))
parser.add_argument("-v", "--verbosity", default=verbosity,
                    type=int, help="0 - silent, 1 - print info")
parser.add_argument("-m", "--exec_mode", default=ExecMode.DEBUG.name,
                    help="choose mode from: " + ",".join(ExecMode.get_names()))
parser.add_argument("-a", "--data_augmentation", default=data_augmentation,
                    type=bool, help="should apply data augmentation, e.g. "
                                    "mirroring, shifting, etc.")
parser.add_argument("-o", "--optimizer_type", default=optimizer_type.name,
                    help="optimizer type from: " + ",".join(
                        OptimizerType.get_names()))


def run():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if mode == ExecMode.DEBUG:
        if verbosity > 0:
            print("DEBUG mode")
        # train on smaller dataste
        limit = debug_limit_tuples
        x_train = x_train[:limit]
        y_train = y_train[:limit]
        x_test = x_test[:limit]
        y_test = y_test[:limit]

    if verbosity > 0:
        print("train size: ", len(y_train))
        print("test size: ", len(y_test))

    if data_augmentation:
        if verbosity > 0:
            print("Applying data augmentation")
        #y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        #y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

    if model_type == DenseNetSize.DENSE_NET_121:
        model = DenseNet121(conv_type=conv_type, classes=num_classes)
    elif model_type == DenseNetSize.DENSE_NET_169:
        model = DenseNet169(conv_type=conv_type, classes=num_classes)
    elif model_type == DenseNetSize.DENSE_NET_201:
        model = DenseNet201(conv_type=conv_type, classes=num_classes)

    batch_denom = 1
    if optimizer_type == OptimizerType.ADAM:
        batch_denom = 100

    learning_rate_fn = learning_rate_with_decay(batch_size=batch_size,
                                                batch_denom=batch_denom,
                                                num_images=len(y_train),
                                                boundary_epochs=[
                                                    int(epochs * 0.5),
                                                    int(epochs * 0.75)],
                                                decay_rates=[1, 0.1, 0.01])

    global_step = tf.train.get_or_create_global_step()
    learning_rate = learning_rate_fn(global_step)

    if optimizer_type is OptimizerType.MOMENTUM:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=0.9)
    elif optimizer_type is OptimizerType.ADAM:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if data_augmentation:
        # compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(x_train)
        # fits the model on batches with real-time data augmentation:
        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) / batch_size,
            epochs=1)

    if verbosity > 0:
        print("epochs: ", epochs)
    # run epoch at a time and then evaluate on the train and test sets
    for epoch in range(epochs):
        if verbosity > 0:
            print("epoch: ", epoch + 1)

        if data_augmentation:
            batches = 0
            for x_batch, y_batch in datagen.flow(x_train, y_train,
                                                 batch_size=batch_size):
                model.fit(x_batch, y_batch, verbose=verbosity)
                batches += 1
                if batches >= len(x_train) / batch_size:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break
        else:
            model.fit(x=x_train, y=y_train, epochs=1, verbose=verbosity,
                      batch_size=batch_size)

        train_eval = model.evaluate(x_train, y_train, verbose=verbosity,
                                    batch_size=batch_size)
        test_eval = model.evaluate(x_test, y_test, verbose=verbosity,
                                   batch_size=batch_size)
        print("evaluation,epoch,", epoch + 1, ",train loss,", train_eval[0],
              ",train accuracy,", train_eval[1], ",test loss,",
              test_eval[0], ",test accuracy,", test_eval[1])

        # the eval return loss for the whole data set (not only a given batch)
        # and the second returned value is the model metric, in this case the
        # accuracy - so we do not need to do it manually as tried below
        # y_test_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
        # y_test_pred = np.argmax(y_test_pred, axis=1)
        # train_accuracy = (len(y_test) - np.count_nonzero(
        #     y_test_pred - np.squeeze(y_test)) + 0.0) / len(y_test)
        # print("train accuracy: ", train_accuracy)


if __name__ == "__main__":
    import sys

    args = parser.parse_args(sys.argv[1:])

    epochs = args.epochs
    debug_limit_tuples = args.debug_limit_tuples
    conv_type = ConvType[args.conv_type]
    verbosity = args.verbosity
    exec_mode = ExecMode[args.exec_mode]
    data_augmentation = args.data_augmentation
    optimizer_type = OptimizerType[args.optimizer_type]
    run()
