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


def _read_image(file_path):
    img = imread(os.path.join(DATA_PATH, file_path.strip()))
    return (img[:, 1:-1, :] / 127.5) - 1


def data_generator(batch_size=64, input_shape=(160, 318, 3), val_set=True):
    """
    Reading data with augmentation
    """
    df = load_dataset()
    df = df[df[VALIDATION_COLUMN] == (1 if val_set else 0)]

    while 1:
        x = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        y = np.zeros((batch_size, 1))
        j = 0

        def add_sample(_img, _steering, i):
            x[i, :, :, :] = _img
            y[i, 0] = _steering
            return i + 1

        while j < batch_size:
            idx = np.random.choice(df.index, 1, replace=False)[0]
            img = _read_image(df.loc[idx, 'center'])
            steering = df.loc[idx, 'steering']

            j = add_sample(img, steering, j)

            if j < batch_size:
                # horizontally flip the image
                j = add_sample(img[:, ::-1, :], -steering, j)

            if j < batch_size and steering < 0:
                # left turn
                j = add_sample(_read_image(df.loc[idx, 'left']), steering * 0.8, j)
                if j < batch_size:
                    j = add_sample(_read_image(df.loc[idx, 'right']), steering * 1.2, j)

            if j < batch_size and steering > 0:
                # right turn
                j = add_sample(_read_image(df.loc[idx, 'right']), steering * 0.8, j)
                if j < batch_size:
                    j = add_sample(_read_image(df.loc[idx, 'left']), steering * 1.2, j)

        yield x, y


def data_generator_original(batch_size=64, input_shape=(160, 318, 3), val_set=True):
    df = load_dataset()
    df = df[df[VALIDATION_COLUMN] == (1 if val_set else 0)]

    while 1:
        x = np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2]))
        y = np.zeros((batch_size, 1))
        for j, idx in enumerate(np.random.choice(df.index, batch_size, replace=False)):
            img = imread(os.path.join(DATA_PATH, df.loc[idx, 'center']))
            img = ((img / 255.) - 0.5) * 2
            x[j, :, :, :] = img[:, 1:-1, :]
            y[j, 0] = df.loc[idx, 'steering']
        yield x, y
