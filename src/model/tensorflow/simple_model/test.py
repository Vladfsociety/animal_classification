import tensorflow as tf
from ..testing import run_test


def test(): 
    model = tf.keras.models.load_model('models/tensorflow/simple_model.keras')    
    run_test(model, "simple_model")

if __name__ == '__main__':
    test()