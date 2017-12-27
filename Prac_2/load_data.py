# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from datetime import datetime
import sklearn.preprocessing as pp


FILE_PATH_TRAIN = "/media/chuntaojun/Mylinux/code/PythonProject/DeepingLearning/sofasofa/Prac_2/data/train.csv"
FILE_PATH_TEST = "/media/chuntaojun/Mylinux/code/PythonProject/DeepingLearning/sofasofa/Prac_2/data/test.csv"


def load_file():
    data = pd.read_csv(FILE_PATH_TRAIN, parse_dates=True)
    data_test = pd.read_csv(FILE_PATH_TEST, parse_dates=True)
    del data['id']
    del data_test['id']
    data = np.array(data.values)
    data_test = np.array(data_test.values)
    return data, data_test


def load_data_train():
    data_set, data_test = load_file()
    train_data, test_data = data_set[0: int(len(data_set) * 0.7), :], data_set[int(len(data_set) * 0.7): len(data_set), :]
    train_x, test_x = [], []
    [train_x.append((datetime.strptime(s, '%Y-%m-%d') - datetime(2000, 1, 1)).days) for s in train_data[:, 0]]
    [test_x.append((datetime.strptime(s, '%Y-%m-%d') - datetime(2000, 1, 1)).days) for s in test_data[:, 0]]
    train_x = np.array(train_x).astype("float64")
    test_x = np.array(test_x).astype("float64")
    train_x = pp.scale(train_x)
    test_x = pp.scale(test_x)
    train_y = pp.minmax_scale(np.array(train_data[:, 1:]).astype("float64"), feature_range=(0, 1))
    test_y = pp.minmax_scale(np.array(test_data[:, 1:]).astype("float64"), feature_range=(0, 1))
    train_x = np.reshape(train_x, (train_x.shape[0], 1, 1))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, 1))
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    load_data_train()
