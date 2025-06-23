import tensorflow as tf
from src.model import UNetPP
from src.predict import predict_single_image

def test_predict_single_image_shape(tmp_path):
    # Create a fake model
    model = UNetPP(input_shape=(64, 64, 3))
    img = tf.random.uniform((64,64,3))
    img_path = tmp_path / "img.png"
    tf.keras.utils.save_img(str(img_path), img)
    mask = predict_single_image(model, str(img_path), 64)
    assert mask.shape == (64, 64, 1) or mask.shape == (64, 64)
