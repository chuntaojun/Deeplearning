# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout
from keras.models import Model
import load_data as ld
import matplotlib.pyplot as plt


def build_model_dense(train_x):
    """

    :param train_x:
    :return:
    """
    inputs = Input(shape=(train_x.shape[1],))
    x_dense_1 = Dense(units=64, activation='relu', use_bias=True)(inputs)
    x_dropout_1 = Dropout(rate=0.1)(x_dense_1)
    x_dense_2 = Dense(units=256, activation='relu', use_bias=True)(x_dropout_1)
    x_dropout_2 = Dropout(rate=0.1)(x_dense_2)
    x_dense_3 = Dense(units=128, activation='relu', use_bias=True)(x_dropout_2)
    predictions = Dense(units=1, activation='sigmoid')(x_dense_3)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def build_model_conv():
    pass


def build_model(data, type='dense'):
    if type == 'dense':
        return build_model_dense(data)
    return build_model_conv(data)


def train_model(batch_size=64, verbose=2, validation_split=0.1, type='dense'):
    """

    :param batch_size:
    :param verbose:
    :param validation_split:
    :param model:
    :param type:
    :return:
    """
    train_name, train_sex, test_name = ld.load_data(type=type)
    model = build_model(train_name, type=type)
    history = model.fit(train_name, train_sex,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        epochs=50,
                        verbose=verbose)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    # score = model.evaluate(test_name, test_sex,
    #                        batch_size=batch_size)
    predictions = model.predict(test_name,
                                batch_size=batch_size,
                                verbose=1)
    return predictions


if __name__ == '__main__':
    import csv
    predictions = train_model(type='dense')
    csvFile = open('test_y.csv', 'w')
    write = csv.writer(csvFile)
    id = 0
    write.writerow(['id', 'gender'])
    for gender in predictions:
        write.writerow([id, int(round(gender))])
        id += 1
