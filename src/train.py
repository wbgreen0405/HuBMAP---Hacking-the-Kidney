import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from src.config import P
from src.losses import dice_coe
from src.augmentations import tf_augment
from src.model import build_unet3plus
from sklearn.model_selection import KFold

def _parse_image_function(example_proto, augment=True):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'mask': tf.io.FixedLenFeature([], tf.string)
    }
    single_example = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape(tf.io.decode_raw(single_example['image'], out_type=tf.uint8), (P['DIM'], P['DIM'], 3))
    mask = tf.reshape(tf.io.decode_raw(single_example['mask'], out_type=tf.bool), (P['DIM'], P['DIM'], 1))
    if augment:
        image, mask = tf_augment(image, mask)
    return tf.cast(image, tf.float32), tf.cast(mask, tf.float32)

def load_dataset(filenames, ordered=False, augment=True):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda ex: _parse_image_function(ex, augment=augment), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

def get_training_dataset(train_filenames):
    dataset = load_dataset(train_filenames)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(128, seed=P['SEED'])
    dataset = dataset.batch(P['BATCH_SIZE'], drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def get_validation_dataset(val_filenames, ordered=True):
    dataset = load_dataset(val_filenames, ordered=ordered, augment=False)
    dataset = dataset.batch(P['BATCH_SIZE'], drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def train_fold(train_filenames, val_filenames, fold, strategy):
    STEPS_PER_EPOCH = len(train_filenames) // P['BATCH_SIZE']
    with strategy.scope():
        model = build_unet3plus(P['DIM'], P['BACKBONE'])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=P['LR']),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[dice_coe, 'accuracy'])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'model-fold-{fold}.h5', monitor='val_dice_coe', save_best_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_dice_coe', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=1e-5)
    history = model.fit(
        get_training_dataset(train_filenames),
        epochs=P['EPOCHS'],
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=get_validation_dataset(val_filenames),
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=P['VERBOSE']
    )
    return model, history