# -*- coding: utf-8 -*-


from keras.layers import Dense, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt
import load_data


def built_model():
    model = Sequential()
    # first layers
    model.add(Dense(units=256,
                    input_dim=2,
                    activation='relu'))
    model.add(Dropout(0.5))

    # second layers
    model.add(Dense(units=128,
                    activation='relu'))
    model.add(Dropout(0.5))

    # third layers
    model.add(Dense(units=1,
                    activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


def train_model(batch_size=32, verbose=2, validation_split=0.2, model=None):
    data, labels = load_data.load_data()
    train_x, train_y = data[: int(len(data) * 0.7), 0: 2], labels[: int(len(labels) * 0.7)]
    test_x, test_y = data[int(len(data) * 0.7): len(data), 0: 2], labels[int(len(labels) * 0.7): len(labels)]
    if model is None:
        model = built_model()
        history = model.fit(train_x, train_y,
                            batch_size=batch_size,
                            epochs=10,
                            verbose=verbose,
                            validation_split=validation_split)
        print "刻画损失函数的变化趋势"
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='valid')
        plt.legend()
        plt.show()
        print "模型构建成功，开始预测数据"
        score = model.evaluate(test_x, test_y,
                               batch_size=batch_size)
        print score
        print "画图"
        predicted = model.predict(test_x,
                                  batch_size=batch_size,
                                  verbose=0)
        rounded = [round(w) for w in predicted]
        plt.scatter(test_y, list(range(len(test_y))), marker='+')
        plt.scatter(rounded, list(range(len(predicted))), marker='*')
        plt.show()


if __name__ == '__main__':
    train_model()
