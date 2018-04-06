# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, Dropout, concatenate
from keras.models import Model
import load_data as ld
import matplotlib.pyplot as plt


def build_model():
    a_name = Input(shape=(24,))
    a_dense_1 = Dense(units=48, activation='relu')(a_name)
    a_dropout = Dropout(rate=0.2)(a_dense_1)
    a_dense_2 = Dense(units=96, activation='relu')(a_dropout)
    encode_1 = Dense(units=24, activation='linear')(a_dense_2)

    b_name = Input(shape=(24,))
    b_dense_1 = Dense(units=48, activation='relu')(b_name)
    b_dropout = Dropout(rate=0.2)(b_dense_1)
    b_dense_2 = Dense(units=96, activation='relu')(b_dropout)
    encode_2 = Dense(units=24, activation='linear')(b_dense_2)

    inputs = concatenate([encode_1, encode_2], axis=-1)
    x_dense_1 = Dense(units=96, activation='relu', use_bias=True)(inputs)
    x_dropout_1 = Dropout(rate=0.1)(x_dense_1)
    x_dense_2 = Dense(units=256, activation='relu', use_bias=True)(x_dropout_1)
    x_dropout_2 = Dropout(rate=0.1)(x_dense_2)
    x_dense_3 = Dense(units=96, activation='relu', use_bias=True)(x_dropout_2)
    predictions = Dense(units=1, activation='sigmoid', name='out_put')(x_dense_3)

    model = Model(inputs=[a_name, b_name], outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def train_model(batch_size=64, verbose=2, validation_split=0.1, type='dense'):
    """

    :param batch_size:
    :param verbose:
    :param validation_split:
    :param type:
    :return:
    """
    train_name, train_sex, test_name, test_sex = ld.load_data(type=type)
    model = build_model()
    history = model.fit([train_name[:, 0: 24], train_name[:, 24:]], train_sex,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        epochs=50,
                        verbose=verbose)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    score = model.evaluate([test_name[:, 0:24], test_name[:, 24:]], test_sex,
                           batch_size=batch_size)
    print(score)


if __name__ == '__main__':
    train_model()
