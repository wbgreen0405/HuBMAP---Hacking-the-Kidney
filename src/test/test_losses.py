import numpy as np
import tensorflow as tf
from src.losses import dice_coe

def test_dice_one():
    y_true = tf.ones((4, 256, 256, 1))
    y_pred = tf.ones((4, 256, 256, 1))
    dice = dice_coe(y_true, y_pred)
    np.testing.assert_almost_equal(dice.numpy(), 1.0, decimal=3)
