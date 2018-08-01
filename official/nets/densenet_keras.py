import argparse
import tensorflow as tf

from official.nets.densenet_model import DenseNet121, DenseNet169, DenseNet201
from official.nets.densenet_model import DenseNetSize
from official.nets.utils import ConvType, EnumWithNames


class ExecMode(EnumWithNames):
    """
    Execution mode.
    """
    DEBUG = 0
    TEST = 1


model_type = DenseNetSize.DENSE_NET_121
epochs = 300
mode = ExecMode.DEBUG
debug_limit_tuples = 256
batch_size = 64
verbosity = 1
conv_type = ConvType.STANDARD
num_classes = 10

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


def run():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    if exec_mode == ExecMode.TEST:
        pass
    elif mode == ExecMode.DEBUG:
        # train on smaller dataste
        limit = debug_limit_tuples
        x_train = x_train[:limit]
        y_train = y_train[:limit]
        x_test = x_test[:limit]
        y_test = y_test[:limit]

    if model_type == DenseNetSize.DENSE_NET_121:
        model = DenseNet121(conv_type=conv_type, classes=num_classes)
    elif model_type == DenseNetSize.DENSE_NET_169:
        model = DenseNet169(conv_type=conv_type, classes=num_classes)
    elif model_type == DenseNetSize.DENSE_NET_201:
        model = DenseNet201(conv_type=conv_type, classes=num_classes)

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if verbosity > 0:
        print("epochs: ", epochs)
    # run epoch at a time and then evaluate on the train and test sets
    for epoch in range(epochs):
        if verbosity > 0:
            print("epoch: ", epoch + 1)
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

    run()
