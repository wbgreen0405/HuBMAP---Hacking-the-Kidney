import tensorflow as tf

def tf_augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.4:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image, k=1)
        mask = tf.image.rot90(mask, k=1)
    if tf.random.uniform(()) > 0.45:
        image = tf.image.random_saturation(image, 0.7, 1.3)
    if tf.random.uniform(()) > 0.45:
        image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, mask
