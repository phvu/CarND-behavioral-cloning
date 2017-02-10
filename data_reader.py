import os

import numpy as np
import pandas as pd
from scipy.misc import imread, imresize

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


def count_dataset(batch_size):
    df = load_dataset()
    valid_size = 1 * np.sum(df[VALIDATION_COLUMN] == 1)
    train_size = (((1 * len(df)) - valid_size) // batch_size) * batch_size
    return train_size, valid_size


def _read_image(file_path):
    img = imread(os.path.join(DATA_PATH, file_path.strip()))
    img = imresize(img, (80, 160, 3))
    return (img / 127.5) - 1


def data_generator(batch_size=64, input_shape=(160, 318, 3), val_set=True):
    df = load_dataset()
    df = df[df[VALIDATION_COLUMN] == (1 if val_set else 0)]

    while 1:
        x = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        y = np.zeros((batch_size, 1))
        j = 0

        while j < batch_size:
            idx = np.random.choice(df.index, batch_size, replace=False)[0]
            file_path = df.loc[idx, 'center']
            file_path = os.path.join(DATA_PATH, file_path[file_path.index('lap00'):])
            img = _read_image(file_path)
            steering = df.loc[idx, 'steering']

            x[j, :, :, :] = img
            y[j, 0] = steering
            j += 1

            # if j < batch_size:
            #     x[j, :, :, :] = img[:, ::-1, :]
            #     y[j, 0] = -steering
            #     j += 1

        yield x, y
