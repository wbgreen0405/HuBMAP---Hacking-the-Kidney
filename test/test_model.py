import tensorflow as tf
from src.model import UNetPP

def test_unetpp_output_shape():
    model = UNetPP(input_shape=(128, 128, 3))
    x = tf.random.normal((2, 128, 128, 3))
    y = model(x)
    assert y.shape == (2, 128, 128, 1)
