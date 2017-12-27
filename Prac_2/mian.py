# -*- coding: utf-8 -*-


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
import load_data as ld
import csv


def f1(y_true, y_pred):
    def question(y_true, y_pred):
        ques = K.mean(K.sum(K.clip(abs(y_true[:, 0] - y_pred[:, 0]) / y_true[:, 0], 0, 1)))
        ans = K.mean(K.sum(K.clip(abs(y_true[:, 1] - y_pred[:, 1]) / y_true[:, 1], 0, 1)))
        return ques, ans

    ques, ans = question(y_true=y_true, y_pred=y_pred)
    return (ques + ans) / 2


def built_model(data):
    model = Sequential()

    # first layer
    model.add(LSTM(units=128,
                   input_shape=(data.shape[1], data.shape[2]),
                   return_sequences=True,
                   activation='tanh'))
    model.add(Dropout(0.3))

    # second layer
    model.add(LSTM(units=128,
                   return_sequences=True,
                   activation='tanh'))
    model.add(Dropout(0.4))

    # third layer
    model.add(LSTM(units=256,
                   return_sequences=False,
                   activation='tanh'))
    model.add(Dense(units=128,
                    activation='relu'))
    model.add(Dropout(0.3))

    # fourth layer
    model.add(Dense(units=2,
                    activation="relu"))
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy', f1])
    model.summary()
    return model


def train_model(batch_size=64, epochs=100, model=None):
    train_x, train_y, test_x, test_y = ld.load_data_train()
    if model is None:
        model = built_model(train_x)
        history = model.fit(train_x, train_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.2,
                            verbose=2)

        print "刻画损失函数的变化趋势"
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='valid')
        plt.legend()
        plt.show()
        print "模型构建成功，开始预测数据"

        predicted = model.predict(test_x,
                                  batch_size=batch_size,
                                  verbose=2)

        plt.plot(test_y, label='true')
        plt.plot(predicted, label='pred')
        plt.legend()
        plt.show()

        return predicted


if __name__ == '__main__':
    predicted = train_model()
    # csvFile = open("test_y.csv", 'w')
    # write = csv.writer(csvFile)
    # write.writerow['id', 'questions', 'answers']
    # id = 2254
    # for ques, ans in predicted:
    #     write.writerow[id, ques, ans]
    #     id += 1
