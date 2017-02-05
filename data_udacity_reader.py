import os
import pandas as pd
import numpy as np
from scipy.ndimage import imread


DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data_udacity'))
VALIDATION_COLUMN = 'valset'
VALIDATION_RATIO = 0.2


def load_dataset():
    log_file = os.path.join(DATA_PATH, 'driving_log.csv')
    log_file_split = os.path.join(DATA_PATH, 'driving_log_split.csv')

    if os.path.exists(log_file_split):
        df = pd.read_csv(log_file_split)
    else:
        df = pd.read_csv(log_file)
        n = len(df)
        print('Dataset has {} samples'.format(n))
        df[VALIDATION_COLUMN] = 1 * (np.random.rand(n) < VALIDATION_RATIO)
        df.to_csv(log_file_split, index=False)
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
            img = imread(os.path.join(DATA_PATH, df.loc[idx, 'center']))
            img = ((img / 255.) - 0.5) * 2
            x[j, :, :, :] = img[:, 1:-1, :]
            y[j, 0] = df.loc[idx, 'steering']
        yield x, y
