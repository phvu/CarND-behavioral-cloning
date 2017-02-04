import os

import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from scipy.ndimage import imread


DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
INPUT_SHAPE = (160, 318, 3)
BATCH_SIZE = 64
EPOCHS = 10
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


def data_generator(val_set=True):
    df = load_dataset()
    df = df[df[VALIDATION_COLUMN] == (1 if val_set else 0), :]

    while 1:
        x = np.zeros((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
        y = np.zeros((BATCH_SIZE, 1))
        for j, idx in enumerate(np.random.choice(len(df), BATCH_SIZE, replace=False)):
            img = imread(os.path.join(DATA_PATH, df.loc[idx, 'center']))
            x[j, :, :, :] = img[:, 1:-1, :]
            y[j, 0] = df.loc[idx, 'steering']
        yield x, y


def create_model():
    img_input = Input(shape=INPUT_SHAPE)

    x = ((img_input / 255.) - 0.5) * 2

    x = Conv2D(128, 4, 4, subsample=(2, 4), bias=False, name='conv1', input_length=BATCH_SIZE)(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation('relu', name='conv1_act')(x)
    x = MaxPooling2D((2, 2))(x)

    # 39 x 39 x 128
    x = Conv2D(256, 3, 3, subsample=(2, 2), bias=False, name='conv2')(x)
    x = BatchNormalization(name='conv2_bn')(x)
    x = Activation('relu', name='conv2_act')(x)
    x = MaxPooling2D((2, 2))(x)

    # 9 x 9 x 256
    x = Conv2D(512, 3, 3, subsample=(2, 2), bias=False, name='conv3')(x)
    x = BatchNormalization(name='conv3_bn')(x)
    x = Activation('relu', name='conv3_act')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)

    # 3 x 3 x 512
    x = Conv2D(1024, 3, 3, subsample=(1, 1), bias=False, name='conv4')(x)
    x = BatchNormalization(name='conv4_bn')(x)
    x = Activation('relu', name='conv4_act')(x)

    # 1 x 1 x 1024
    x = Flatten(name='flatten')(x)
    x = Dense(1024, name='ff1')(x)
    x = BatchNormalization(name='ff1_bn')(x)
    x = Activation('relu', name='ff1_act')(x)

    x = Dense(1024, name='ff2')(x)
    x = BatchNormalization(name='ff2_bn')(x)
    x = Activation('relu', name='ff2_act')(x)

    predictions = Dense(1, activation='linear', name='predictions')(x)

    return Model(input=img_input, output=predictions)


def train():
    m = create_model()

    optimizer = SGD(lr=0.01, momentum=0.9, decay=0.8, nesterov=False)
    m.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[])

    cnts = count_dataset()
    print('Training size: {}, Validation size: {}'.format(*cnts))

    history = m.fit_generator(data_generator(val_set=False), samples_per_epoch=cnts[0], nb_epoch=EPOCHS,
                              verbose=1, validation_data=data_generator(val_set=True), pickle_safe=True)

    score = m.evaluate_generator(data_generator(val_set=True), val_samples=cnts[1], pickle_safe=True)
    print('Validation MSE:', score[0])
    m.save('model.h5')

    return m, history

if __name__ == '__main__':
    train()
