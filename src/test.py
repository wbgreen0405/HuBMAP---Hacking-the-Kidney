import numpy as np
import pandas as pd
import os
import glob
import gc

import rasterio
from rasterio.windows import Window

import pathlib
from tqdm.notebook import tqdm
import cv2

import tensorflow as tf
import efficientnet as efn
import efficientnet.tfkeras


def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2 
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx,ny, 4), dtype=np.int64)
    
    for i in range(nx):
        for j in range(ny):
            slices[i,j] = x1[i], x2[i], y1[j], y2[j]    
    return slices.reshape(nx*ny,4)
    
    
    
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
fold_models = []
for fold_model_path in glob.glob(mod_path+'*.h5'):
    fold_models.append(tf.keras.models.load_model(fold_model_path,compile = False))
print(len(fold_models))

AUTO = tf.data.experimental.AUTOTUNE
image_feature = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'x1': tf.io.FixedLenFeature([], tf.int64),
    'y1': tf.io.FixedLenFeature([], tf.int64)
}
def _parse_image(example_proto):
    example = tf.io.parse_single_example(example_proto, image_feature)
    image = tf.reshape( tf.io.decode_raw(example['image'],out_type=np.dtype('uint8')), (P['DIM'],P['DIM'], 3))
    return image, example['x1'], example['y1']

def load_dataset(filenames, ordered=True):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(_parse_image)
    return dataset

def get_dataset(FILENAME):
    dataset = load_dataset(FILENAME)
    dataset  = dataset.batch(64)
    dataset = dataset.prefetch(AUTO)
    return dataset
    
TTAS = [0, 1, 2]

def flip(image, axis=0):
    if axis == 1:
        return image[::-1, :, ]
    elif axis == 2:
        return image[:, ::-1, ]
    elif axis == 3:
        return image[::-1, ::-1, ]
    else:
        return image
        
p = pathlib.Path('../input/hubmap-kidney-segmentation')
subm = {}

for i, filename in tqdm(enumerate(p.glob('test/*.tiff')), 
                        total = len(list(p.glob('test/*.tiff')))):
    
    print(f'{i+1} Predicting {filename.stem}')
    
    dataset = rasterio.open(filename.as_posix(), transform = identity)
    preds = np.zeros(dataset.shape, dtype=np.uint8)    
    
    if SUBMISSION_MODE == 'PUBLIC_TFREC' and MIN_OVERLAP == 300 and WINDOW == 1024 and NEW_SIZE == 512:
        print('SUBMISSION_MODE: PUBLIC_TFREC')
        fnames = glob.glob('/kaggle/input/hubmap-tfrecords-1024-512-test/test/'+filename.stem+'*.tfrec')
        
        if len(fnames)>0: # PUBLIC TEST SET
            for FILENAME in fnames:
                pred = None
                for fold_model in fold_models:
                    tmp = fold_model.predict(get_dataset(FILENAME))/len(fold_models)
                    if pred is None:
                        pred = tmp
                    else:
                        pred += tmp
                    del tmp
                    gc.collect()

                pred = tf.cast((tf.image.resize(pred, (WINDOW,WINDOW)) > THRESHOLD),tf.bool).numpy().squeeze()

                idx = 0
                for img, X1, Y1 in get_dataset(FILENAME):
                    for fi in range(X1.shape[0]):
                        x1 = X1[fi].numpy()
                        y1 = Y1[fi].numpy()
                        preds[x1:(x1+WINDOW),y1:(y1+WINDOW)] += pred[idx]
                        idx += 1
                        
        else: # IGNORE PRIVATE TEST SET (CREATE TFRECORDS IN FUTURE)
            pass
    else:
        print('SUBMISSION_MODE: FULL')
        slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)

        if dataset.count != 3:
            print('Image file with subdatasets as channels')
            layers = [rasterio.open(subd) for subd in dataset.subdatasets]
            
        for (x1,x2,y1,y2) in slices:
            if dataset.count == 3:
                image = dataset.read([1,2,3],
                            window=Window.from_slices((x1,x2),(y1,y2)))
                image = np.moveaxis(image, 0, -1)
            else:
                image = np.zeros((WINDOW, WINDOW, 3), dtype=np.uint8)
                for fl in range(3):
                    image[:,:,fl] = layers[fl].read(window=Window.from_slices((x1,x2),(y1,y2)))
                    
            image = cv2.resize(image, (NEW_SIZE, NEW_SIZE),interpolation = cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = np.expand_dims(image, 0)

            pred = None

            for fold_model in fold_models:
                if pred is None:
                    pred = np.squeeze(fold_model.predict(image))
                else:
                    pred += np.squeeze(fold_model.predict(image))

            pred = pred/len(fold_models)

            pred = cv2.resize(pred, (WINDOW, WINDOW))
            preds[x1:x2,y1:y2] += (pred > THRESHOLD).astype(np.uint8)

    preds = (preds > 0.5).astype(np.uint8)
    
    subm[i] = {'id':filename.stem, 'predicted': rle_encode_less_memory(preds)}
    
    if CHECKSUM:
        print('Checksum: '+ str(np.sum(preds)))
    
    del preds
    gc.collect();


p = pathlib.Path('../input/hubmap-kidney-segmentation')
subm = {}

for i, filename in tqdm(enumerate(p.glob('test/*.tiff')), 
                        total = len(list(p.glob('test/*.tiff')))):
    
    print(f'{i+1} Predicting {filename.stem}')
    
    dataset = rasterio.open(filename.as_posix(), transform = identity)
    preds = np.zeros(dataset.shape, dtype=np.uint8)    
    
    if SUBMISSION_MODE == 'PUBLIC_TFREC' and MIN_OVERLAP == 300 and WINDOW == 1024 and NEW_SIZE == 512:
        print('SUBMISSION_MODE: PUBLIC_TFREC')
        fnames = glob.glob('/kaggle/input/hubmap-tfrecords-1024-512-test/test/'+filename.stem+'*.tfrec')
        
        if len(fnames)>0: # PUBLIC TEST SET
            for FILENAME in fnames:
                pred = None
                for fold_model in fold_models:
                    tmp = fold_model.predict(get_dataset(FILENAME))/len(fold_models)
                    if pred is None:
                        pred = tmp
                    else:
                        pred += tmp
                    del tmp
                    gc.collect()

                pred = tf.cast((tf.image.resize(pred, (WINDOW,WINDOW)) > THRESHOLD),tf.bool).numpy().squeeze()

                idx = 0
                for img, X1, Y1 in get_dataset(FILENAME):
                    for fi in range(X1.shape[0]):
                        x1 = X1[fi].numpy()
                        y1 = Y1[fi].numpy()
                        preds[x1:(x1+WINDOW),y1:(y1+WINDOW)] += pred[idx]
                        idx += 1
                        
        else: # IGNORE PRIVATE TEST SET (CREATE TFRECORDS IN FUTURE)
            pass
    else:
        print('SUBMISSION_MODE: FULL')
        slices = make_grid(dataset.shape, window=WINDOW, min_overlap=MIN_OVERLAP)

        if dataset.count != 3:
            print('Image file with subdatasets as channels')
            layers = [rasterio.open(subd) for subd in dataset.subdatasets]
            
        for (x1,x2,y1,y2) in slices:
            if dataset.count == 3:
                image = dataset.read([1,2,3],
                            window=Window.from_slices((x1,x2),(y1,y2)))
                image = np.moveaxis(image, 0, -1)
            else:
                image = np.zeros((WINDOW, WINDOW, 3), dtype=np.uint8)
                for fl in range(3):
                    image[:,:,fl] = layers[fl].read(window=Window.from_slices((x1,x2),(y1,y2)))
                    
            image = cv2.resize(image, (NEW_SIZE, NEW_SIZE),interpolation = cv2.INTER_AREA)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = np.expand_dims(image, 0)

            pred = None

            for fold_model in fold_models:
                if pred is None:
                    pred = np.squeeze(fold_model.predict(image))
                else:
                    pred += np.squeeze(fold_model.predict(image))

            pred = pred/len(fold_models)

            pred = cv2.resize(pred, (WINDOW, WINDOW))
            preds[x1:x2,y1:y2] += (pred > THRESHOLD).astype(np.uint8)

    preds = (preds > 0.5).astype(np.uint8)
    
    subm[i] = {'id':filename.stem, 'predicted': rle_encode_less_memory(preds)}
    
    if CHECKSUM:
        print('Checksum: '+ str(np.sum(preds)))
    
    del preds
    gc.collect();
    
    
submission = pd.DataFrame.from_dict(subm, orient='index')
submission.to_csv('submission.csv', index=False)
submission.head()



