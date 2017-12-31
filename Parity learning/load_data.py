# -*- coding: utf-8 -*-


import numpy as np
import sklearn.preprocessing as pp


def load_data(number=10001):
    temp = [i for i in range(1, number * 2)]
    data, labels = [], []
    for i in temp:
        if i % 2 == 0:
            data.append([i, i % 10, 1])
        else:
            data.append([i, i % 10, 0])
    print "归一化之前：\n", np.array(data)
    seclar = pp.MinMaxScaler(feature_range=(0, 1))
    data = seclar.fit_transform(data)
    np.random.shuffle(data)
    print "归一化之后：\n", data
    labels = data[:, -1]
    return data, labels


if __name__ == '__main__':
    load_data()
