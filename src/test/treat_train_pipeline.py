import pytest
import os
import numpy as np
from src.train import load_image, load_mask

def test_load_image_and_mask_shapes(tmp_path):
    # Create fake image and mask
    img = (np.ones((256,256,3))*255).astype(np.uint8)
    mask = (np.ones((256,256,1))*255).astype(np.uint8)
    img_path = tmp_path / "img.png"
    mask_path = tmp_path / "mask.png"
    import cv2
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(mask_path), mask)
    out_img = load_image(str(img_path), 256)
    out_mask = load_mask(str(mask_path), 256)
    assert out_img.shape == (256,256,3)
    assert out_mask.shape == (256,256,1)
