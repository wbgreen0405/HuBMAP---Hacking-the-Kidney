import yaml

P = {
    'EPOCHS': 30,
    'BACKBONE': 'EfficientNetB0',
    'NFOLDS': 5,
    'SEED': 0,
    'VERBOSE': 0,
    'DISPLAY_PLOT': True,
    'BATCH_COE': 8,
    'TILING': [1024,512],
    'DIM': 512,
    'DIM_FROM': 1024,
    'LR': 5e-4,
    'OVERLAPP': True,
    'STEPS_COE': 1,
}

P['DIM'] = P['TILING'][1]
P['DIM_FROM'] = P['TILING'][0]

with open('params.yaml', 'w') as file:
    yaml.dump(P, file)
