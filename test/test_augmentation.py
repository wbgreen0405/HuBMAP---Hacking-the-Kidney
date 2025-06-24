import numpy as np
import tensorflow as tf
from src.augmentations import tf_augment

def test_tf_augment_shapes():
    img = tf.ones((256, 256, 3), dtype=tf.float32)
    mask = tf.ones((256, 256, 1), dtype=tf.float32)
    aug_img, aug_mask = tf_augment(img, mask)
    assert aug_img.shape == (256, 256, 3)
    assert aug_mask.shape == (256, 256, 1)
