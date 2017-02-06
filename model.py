import argparse
import os

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Reshape, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Nadam

# from data_udacity_reader import data_generator, count_dataset
from data_reader import data_generator, count_dataset


def create_model(input_shape=(160, 318, 3)):
    img_input = Input(shape=input_shape)

    x = Conv2D(256, 3, 3, subsample=(1, 2), bias=False, name='conv0')(img_input)
    # x = BatchNormalization(name='conv0_bn')(x)
    x = Activation('relu', name='conv0_act')(x)
    x = MaxPooling2D((2, 2))(x)

    # 79 x 79 x 256
    x = Conv2D(256, 3, 3, subsample=(2, 2), bias=False, name='conv1')(img_input)
    # x = BatchNormalization(name='conv1_bn')(x)
    x = Activation('relu', name='conv1_act')(x)
    x = MaxPooling2D((2, 2))(x)

    # 19 x 19 x 256
    x = Conv2D(512, 3, 3, subsample=(2, 2), bias=False, name='conv2')(x)
    # x = BatchNormalization(name='conv2_bn')(x)
    x = Activation('relu', name='conv2_act')(x)
    x = MaxPooling2D((2, 2))(x)

    # 4 x 4 x 512
    x = Conv2D(1024, 2, 2, subsample=(2, 2), bias=False, name='conv3')(x)
    # x = BatchNormalization(name='conv3_bn')(x)
    x = Activation('relu', name='conv3_act')(x)
    x = MaxPooling2D((2, 2), strides=(1, 1))(x)

    # 1 x 1 x 1024
    x = Reshape((1024,))(x)
    x = Dense(1024, name='ff1')(x)
    # x = BatchNormalization(name='ff1_bn')(x)
    x = Activation('relu', name='ff1_act')(x)

    x = Dense(1024, name='ff2')(x)
    # x = BatchNormalization(name='ff2_bn')(x)
    x = Activation('relu', name='ff2_act')(x)

    predictions = Dense(1, activation='linear', name='predictions')(x)

    return Model(input=img_input, output=predictions)


def train(model_path='model.h5'):
    epochs = 10
    batch_size = 64
    input_shape = (160, 318, 3)

    m = create_model(input_shape=input_shape)

    # optimizer = SGD(lr=0.01, momentum=0.9, decay=0.8, nesterov=False)
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
    m.save(model_path)

    return m, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition h5 to be saved')
    args = parser.parse_args()

    train(args.model)
