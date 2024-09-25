import tensorflow as tf
from ..testing import run_test


def test():
    model = tf.keras.models.load_model('models/tensorflow/vgg11.keras')
    run_test(model, "vgg11")

if __name__ == '__main__':
    test()