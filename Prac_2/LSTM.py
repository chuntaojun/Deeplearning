# -*- coding: utf-8 -*-


from keras.layers import Input, LSTM, Dense, Dropout
from keras import Model
import matplotlib.pyplot as plt
import load_data as ld


def built_model(data):
    inputs = Input(shape=(data.shape[1], data.shape[2]))

    x_lstm_1 = LSTM(units=128,
                    return_sequences=True,
                    activation='tanh')(inputs)
    x_dropout_1 = Dropout(0.2)(x_lstm_1)
    x_lstm_2 = LSTM(units=256,
                    return_sequences=True,
                    activation='tanh')(x_dropout_1)
    x_dropout_2 = Dropout(0.2)(x_lstm_2)
    x_lstm_3 = LSTM(units=256,
                    return_sequences=False,
                    activation='tanh')(x_dropout_2)
    x_dense_1 = Dense(units=128,
                      use_bias=True,
                      activation='relu')(x_lstm_3)
    y = Dense(units=2,
              use_bias=True,
              activation="relu")(x_dense_1)
    model = Model(inputs=inputs, outputs=y)
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mape'])
    model.summary()
    return model


def train_model(batch_size=64, epochs=25, model=None):
    train_x, train_y, test_x, test_y = ld.train_test_data(ld.load_data_train())
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
