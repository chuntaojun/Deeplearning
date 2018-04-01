# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import sklearn.preprocessing as scaler


FILE_PATH_TRAIN = "data/train.csv"
FILE_PATH_TEST = "data/test.csv"


def load_file():
    data = pd.read_csv(FILE_PATH_TRAIN, parse_dates=True)
    data_test = pd.read_csv(FILE_PATH_TEST, parse_dates=True)
    return data, data_test


def load_data_train(type='deep'):
    data_set, data_test = load_file()
    if type is not 'deep':
        return data_set.values
    else:
        train = [data_set['questions'], data_set['answers']]
        return np.array(train).transpose()


def train_test_data(data):
    """

    :param data:
    :return:
    """
    min_max_scaler = scaler.MinMaxScaler(feature_range=(0, 1))

    def create_data_set(data_set):
        data_x, data_y = [], []
        for i in range(len(data_set) - 2):
            data_x.append(np.ndarray.flatten(data_set[i: i + 2, :]))
            data_y.append(data_set[i + 2, :])
        return np.array(data_x).astype('float64'), np.array(data_y).astype('float64')

    train_x, train_y = create_data_set(min_max_scaler.fit_transform(data[: int(len(data) * 0.9), :]))
    test_x, test_y = create_data_set(min_max_scaler.fit_transform(data[int(len(data) * 0.9): len(data), :]))
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression
    # 读取数据
    train = pd.read_csv(FILE_PATH_TRAIN)
    test = pd.read_csv(FILE_PATH_TEST)
    # submit = pd.read_csv("sample_submit.csv")

    # 取出真实值：questions和answers
    q_train = train.pop('questions')
    a_train = train.pop('answers')

    # 把date转为时间格式，得到星期，再进行独热处理
    train['date'] = pd.to_datetime(train['date'])
    train['dayofweek'] = train['date'].dt.dayofweek
    train = pd.get_dummies(train, columns=['dayofweek'])
    test['date'] = pd.to_datetime(test['date'])
    test['dayofweek'] = test['date'].dt.dayofweek
    test = pd.get_dummies(test, columns=['dayofweek'])

    # 插入id与星期的交叉相，一共得到7项
    for i in range(7):
        train['id_dayofweek_%s' % i] = train['id'] * train['dayofweek_%s' % i]
        test['id_dayofweek_%s' % i] = test['id'] * test['dayofweek_%s' % i]

    # 去掉date这一列
    train.drop('date', axis=1, inplace=True)
    test.drop('date', axis=1, inplace=True)

    # 建立多变量线性回归模型并进行预测

    # 预测questions
    reg = LinearRegression()
    reg.fit(train, q_train)
    q_pred = reg.predict(test)

    # 预测answers
    reg = LinearRegression()
    reg.fit(train, a_train)
    a_pred = reg.predict(test)
