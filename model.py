from keras.layers import Dense, Activation, Reshape, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD

# from data_udacity_reader import data_generator, count_dataset
from data_reader import data_generator, count_dataset


def create_model(input_shape=(160, 318, 3)):
    img_input = Input(shape=input_shape)

    x = Conv2D(128, 4, 4, subsample=(2, 4), bias=False, name='conv1')(img_input)
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
    x = Reshape(1024)(x)
    x = Dense(1024, name='ff1')(x)
    x = BatchNormalization(name='ff1_bn')(x)
    x = Activation('relu', name='ff1_act')(x)

    x = Dense(1024, name='ff2')(x)
    x = BatchNormalization(name='ff2_bn')(x)
    x = Activation('relu', name='ff2_act')(x)

    predictions = Dense(1, activation='linear', name='predictions')(x)

    return Model(input=img_input, output=predictions)


def train():
    epochs = 10
    batch_size = 64
    input_shape = (160, 318, 3)

    m = create_model(input_shape=input_shape)

    optimizer = SGD(lr=0.01, momentum=0.9, decay=0.8, nesterov=False)
    m.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[])

    cnts = count_dataset()
    print('Training size: {}, Validation size: {}'.format(*cnts))

    history = m.fit_generator(data_generator(batch_size=batch_size, input_shape=input_shape, val_set=False),
                              samples_per_epoch=cnts[0], nb_epoch=epochs, verbose=1,
                              validation_data=data_generator(batch_size=batch_size,
                                                             input_shape=input_shape, val_set=True),
                              nb_val_samples=cnts[1], pickle_safe=True)

    score = m.evaluate_generator(data_generator(batch_size=batch_size, input_shape=input_shape, val_set=True),
                                 val_samples=cnts[1], pickle_safe=True)
    print('Validation MSE:', score)
    m.save('model.h5')

    return m, history


if __name__ == '__main__':
    train()
