# -*- coding: utf-8 -*-


"""
训练集中共有4000个灰度图像，预测集中有3550个灰度图像。每个图像中都会含有大量的噪点。
图像的分辨率为40x40，也就是40x40的矩阵，每个矩阵以行向量的形式被存放在train.csv和test.csv中。train.csv和test.csv中每行数据代表一个图像，
也就是说每行都有1600个特征。
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


FILE_PATH_train = "/media/chuntaojun/Mylinux/code/PythonProject/DeepingLearning/sofasofa/Prac_1/data/train.csv"
FILE_PATH_test = "/media/chuntaojun/Mylinux/code/PythonProject/DeepingLearning/sofasofa/Prac_1/data/test.csv"


def load_train_test_data():
    data_train = pd.read_csv(FILE_PATH_train)
    data_test = pd.read_csv(FILE_PATH_test)
    del data_train['id']
    del data_test['id']
    data_train = np.array(data_train.values)
    np.random.shuffle(data_train)
    labels = data_train[:, -1]
    data_test = np.array(data_test.values)
    data, data_test = data_modify_suitable_train(data_train, True), data_modify_suitable_train(np.array(data_test),
                                                                                               False)
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.7)
    return train_x, train_y, test_x, test_y, data_test


def data_modify_suitable_train(data_set=None, type=True):
    if data_set is not None:
        data = []
        if type is True:
            np.random.shuffle(data_set)
            data = data_set[:, 0: data_set.shape[1] - 1]
        else:
            data = data_set
        data = np.array([np.reshape(i, (40, 40)) for i in data])
        # 将矩阵重新塑形为 （40， 40， 1）的形状，以符合卷积神经网络的输入
        data = np.array([np.reshape(i, (i.shape[0], i.shape[1], 1)) for i in data])
        return data


if __name__ == '__main__':
    load_train_test_data()
