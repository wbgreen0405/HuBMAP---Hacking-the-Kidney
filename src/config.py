import os

def get_settings(batch_size=6, epochs=2, encoder='resnet34'):
    return {
        'BATCH_SIZE': batch_size,
        'EPOCHS': epochs,
        'ENCODER': encoder
    }

DATA_ORIG_PATH = '../input/hubmap-kidney-segmentation'
DATA_PATH = '../input/hubmap-256x256'
PATH_TRAIN = os.path.join(DATA_PATH, 'train')
PATH_MASKS = os.path.join(DATA_PATH, 'masks')
PATH_SUBMISSION = os.path.join(DATA_ORIG_PATH, 'sample_submission.csv')
