# -*- coding: utf-8 -*-
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras import Model
from keras import backend as K
from keras.callbacks import Callback
import tensorflow as tf

from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True)


class PRAucEvaluation(Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(PRAucEvaluation, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if (self.include_on_batch):
            logs['pr_auc_val'] = float('-inf')
            if (self.validation_data):
                logs['pr_auc_val'] = average_precision_score(self.validation_data[1],
                                                             self.model.predict(self.validation_data[0],
                                                                                batch_size=self.predict_batch_size))

    def on_train_begin(self, logs={}):
        if not ('pr_auc_val' in self.params['metrics']):
            self.params['metrics'].append('pr_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['pr_auc_val'] = float('-inf')
        if (self.validation_data):
            logs['pr_auc_val'] = average_precision_score(self.validation_data[1],
                                                         self.model.predict(self.validation_data[0],
                                                                            batch_size=self.predict_batch_size))


class DNN_Model(object):
    def __init__(self, data_x, batch_size):
        self.shape = data_x.shape[1]
        self.batch_size = batch_size

    def build_model(self, loss='binary_crossentropy'):
        x_input = Input(shape=(self.shape,))
        x_dense_1 = Dense(units=64, activation='relu')(x_input)
        x_dense_2 = Dense(units=128, activation='relu')(x_dense_1)
        # x_dropout_1 = Dropout(rate=0.2)(x_dense_2)
        x_bn_1 = BatchNormalization()(x_dense_2)
        x_dense_3 = Dense(units=256, activation='relu')(x_bn_1)
        # x_dropout_2 = Dropout(rate=0.1)(x_dense_3)
        x_bn_2 = BatchNormalization()(x_dense_3)
        x_dense_4 = Dense(units=128, activation='relu')(x_bn_2)
        x_bn_3 = BatchNormalization()(x_dense_4)
        x_dense_5 = Dense(units=32, activation='relu')(x_bn_3)
        y = Dense(units=1, activation='sigmoid')(x_dense_5)

        model = Model(inputs=x_input, outputs=y)
        model.compile(loss=loss,
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        self.model = model

    def train_model(self, x, y, callback, epochs=30):
        if self.model is None:
            raise Exception("Please run obj.build_model() function")
        history = self.model.fit(x=x, y=y,
                                 batch_size=self.batch_size,
                                 epochs=epochs,
                                 verbose=2,
                                 validation_split=0.1,
                                 callbacks=[callback])
        self.model.save("prac_4.h5")
        self.history = history

    def paint(self):
        print "刻画损失函数在训练与验证集的变化"
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='valid')
        plt.legend()
        plt.show()

    def prediction(self, x):
        predicted = self.model.predict(x=x,
                                       batch_size=self.batch_size,
                                       verbose=2)
        return np.array(predicted)

    def score(self, x_t, y_t):
        score = self.model.evaluate(x_t, y_t,
                                    batch_size=self.batch_size)
        print(score)


if __name__ == '__main__':
    import load_file
    import write_file

    LoadData = load_file.LoadData()

    train_x, train_y, test_x, test_y = LoadData.train_test_data()

    dnn = DNN_Model(data_x=train_x, batch_size=64)
    dnn.build_model()
    PRAuc = PRAucEvaluation()
    dnn.train_model(train_x, train_y, callback=PRAuc, epochs=50)
    # dnn.paint()
    predictions = dnn.prediction(x=test_x)
    print(average_precision_score(y_true=test_y, y_score=predictions.flatten()))
    # write2file = write_file.Write_Csv()
    # write2file.write_file(predictions=predictions, title=['CaseId', 'Evaluation'], id=200001)
