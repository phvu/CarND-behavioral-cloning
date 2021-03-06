import argparse
import os

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Reshape, Conv2D, MaxPooling2D, Input
from keras.models import Model
from keras.optimizers import Nadam

from data_reader import data_generator, count_dataset


def create_model(input_shape=(80, 160, 3)):
    img_input = Input(shape=input_shape)

    x = Conv2D(32, 3, 3, subsample=(1, 2), bias=False, name='conv0')(img_input)
    x = Activation('relu', name='conv0_act')(x)
    x = MaxPooling2D((2, 2))(x)

    # 39 x 39 x 32
    x = Conv2D(64, 3, 3, subsample=(2, 2), bias=False, name='conv1')(x)
    x = Activation('relu', name='conv1_act')(x)
    x = MaxPooling2D((2, 2))(x)

    # 9 x 9 x 64
    x = Conv2D(64, 3, 3, subsample=(2, 2), bias=False, name='conv2')(x)
    x = Activation('relu', name='conv2_act')(x)
    x = MaxPooling2D((2, 2))(x)

    # 2 x 2 x 64
    x = Conv2D(128, 2, 2, subsample=(2, 2), bias=False, name='conv3')(x)
    x = Activation('relu', name='conv3_act')(x)

    # 1 x 1 x 128
    x = Reshape((128,))(x)
    x = Dense(128, name='ff1')(x)
    x = Activation('relu', name='ff1_act')(x)

    x = Dense(128, name='ff2')(x)
    x = Activation('relu', name='ff2_act')(x)

    predictions = Dense(1, activation='linear', name='predictions')(x)

    return Model(input=img_input, output=predictions)


def train(model_path='model.h5'):
    epochs = 10
    batch_size = 64
    input_shape = (80, 160, 3)

    m = create_model(input_shape=input_shape)

    optimizer = Nadam()
    m.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[])

    cnts = count_dataset(batch_size)
    print('Training size: {}, Validation size: {}'.format(*cnts))

    checkpointer = ModelCheckpoint(filepath=os.path.join(os.path.split(__file__)[0], model_path),
                                   verbose=1, save_best_only=True)

    history = m.fit_generator(data_generator(batch_size=batch_size, input_shape=input_shape, val_set=False),
                              samples_per_epoch=cnts[0], nb_epoch=epochs, verbose=1,
                              validation_data=data_generator(batch_size=batch_size,
                                                             input_shape=input_shape, val_set=True),
                              nb_val_samples=cnts[1], pickle_safe=True,
                              callbacks=[checkpointer])

    score = m.evaluate_generator(data_generator(batch_size=batch_size, input_shape=input_shape, val_set=True),
                                 val_samples=cnts[1], pickle_safe=True)
    print('Validation MSE:', score)

    return m, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition h5 to be saved')
    args = parser.parse_args()

    train(args.model)
