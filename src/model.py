from keras_unet_collection import models

def build_unet3plus(dim, backbone, weights='imagenet'):
    model = models.unet_3plus_2d(
        (dim, dim, 3), n_labels=1, 
        filter_num_down=[64, 128, 256, 512, 1024],
        filter_num_skip='auto', filter_num_aggregate='auto',
        activation='GELU', output_activation='Sigmoid', batch_norm=True,
        backbone=backbone, weights=weights, 
        deep_supervision=True, pool=True, unpool=True, name='unet3plus'
    )
    return model
