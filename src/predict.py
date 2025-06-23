import numpy as np
import cv2
import tensorflow as tf
import rasterio
from rasterio.windows import Window
from src.model import build_unet3plus

def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def make_grid(shape, window=256, min_overlap=32):
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx, ny, 4), dtype=np.int64)
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]
    return slices.reshape(nx * ny, 4)

def predict_slide(models, tiff_path, window, min_overlap, new_size):
    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
    dataset = rasterio.open(tiff_path, transform=identity)
    slices = make_grid(dataset.shape, window=window, min_overlap=min_overlap)
    preds = np.zeros(dataset.shape, dtype=np.uint8)
    for (x1, x2, y1, y2) in slices:
        image = dataset.read([1, 2, 3], window=Window.from_slices((x1, x2), (y1, y2)))
        image = np.moveaxis(image, 0, -1)
        image = cv2.resize(image, (new_size, new_size))
        image = np.expand_dims(image, 0)
        pred = np.mean([np.squeeze(model.predict(image)) for model in models], axis=0)
        pred = cv2.resize(pred, (window, window))
        preds[x1:x2, y1:y2] = (pred > 0.5).astype(np.uint8)
    return preds

# Usage: load models, call predict_slide(models, ...)
