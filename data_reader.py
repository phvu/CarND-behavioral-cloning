import os

import numpy as np
import pandas as pd
from scipy.ndimage import imread

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
VALIDATION_COLUMN = 'valset'
VALIDATION_RATIO = 0.2


def load_dataset():
    log_file = os.path.join(DATA_PATH, 'driving_log.csv')

    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
    else:
        df = None
        for sub_dir in ('lap001', 'lap002', 'lap003', 'lap004'):
            df_sub = pd.read_csv(os.path.join(DATA_PATH, sub_dir, 'driving_log.csv'), header=None,
                                 names=['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'])
            if df is None:
                df = df_sub
            else:
                df = df.append(df_sub, ignore_index=True)
        n = len(df)
        print('Dataset has {} samples'.format(n))
        df[VALIDATION_COLUMN] = 1 * (np.random.rand(n) < VALIDATION_RATIO)
        df.to_csv(log_file, index=False)
    return df


def count_dataset():
    df = load_dataset()
    valid_size = np.sum(df[VALIDATION_COLUMN] == 1)
    return len(df) - valid_size, valid_size


def data_generator(batch_size=64, input_shape=(160, 318, 3), val_set=True):
    df = load_dataset()
    df = df[df[VALIDATION_COLUMN] == (1 if val_set else 0), :]

    while 1:
        x = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        y = np.zeros((batch_size, 1))
        for j, idx in enumerate(np.random.choice(len(df), batch_size, replace=False)):
            file_path = df.loc[idx, 'center']
            file_path = os.path.join(DATA_PATH, file_path[file_path.index('lap00'):])
            img = imread(file_path)
            img = ((img / 255.) - 0.5) * 2
            x[j, :, :, :] = img[:, 1:-1, :]
            y[j, 0] = df.loc[idx, 'steering']
        yield x, y
