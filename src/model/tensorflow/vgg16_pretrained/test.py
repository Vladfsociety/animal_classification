import tensorflow as tf
from ..testing import run_test


def test():
    model = tf.keras.models.load_model('models/tensorflow/vgg16_pretrained.keras')
    run_test(model, "vgg16_pretrained")

if __name__ == '__main__':
    test()