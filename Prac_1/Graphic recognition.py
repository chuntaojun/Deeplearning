# -*- coding: utf-8 -*-
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, MaxPooling2D, Flatten, Convolution2D
from keras.models import Sequential
from keras import backend as K
import load_data as ld
import matplotlib.pyplot as plt
import csv
import numpy as np


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def built_model():
    model = Sequential()

    # first layers
    model.add(Convolution2D(filters=8,
                            kernel_size=(5, 5),
                            input_shape=(40, 40, 1),
                            activation='relu'))
    model.add(Convolution2D(filters=16,
                            kernel_size=(3, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second layers
    model.add(Convolution2D(filters=16,
                            kernel_size=(3, 3),
                            activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # third layers
    model.add(Dense(units=128,
                    activation='relu'))
    model.add(Dropout(0.5))

    # fourth layers
    model.add(Dense(units=1,
                    activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', f1])

    model.summary()
    return model


def train_model(batch_size=64, epochs=20, model=None):
    train_x, train_y, test_x, test_y, t = ld.load_train_test_data()
    if model is None:
        model = built_model()
        history = model.fit(train_x, train_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            validation_split=0.1,
                            callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        print "刻画损失函数在训练与验证集的变化"
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='valid')
        plt.legend()
        plt.show()

        predicted = model.predict(t,
                                  batch_size=batch_size,
                                  verbose=1)
        predicted = np.array([round(w) for w in predicted])

        score = model.evaluate(test_x, test_y,
                               batch_size=batch_size)
        print score

        print "刻画预测结果与测试集结果"

        # count = 0
        # for i in range(len(predicted)):
        #     print [predicted[i], test_y[i]]
        #     if predicted[i] == test_y[i]:
        #         count += 1
        # print "正确个数：" + str(count)
        # print "正确率：" + str(count * 1.0 /len(predicted))
        model.save('my_model.h5')
        return predicted


if __name__ == '__main__':
    predicted = train_model()
    num = 4000
    csvFile = open('test_y.csv', 'w')
    write = csv.writer(csvFile)
    write.writerow(['id', 'y'])
    for i in predicted:
        write.writerow([num, int(i)])
        num += 1
    print predicted
