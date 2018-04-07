# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as scale
import sklearn.decomposition.pca as pca


class LoadData(object):
    def __init__(self):
        self.file_train = '/Volumes/Extended/code/PythonProject/DeepingLearning/sofasofa/prac_4/data/train.csv'
        self.file_test = '/Volumes/Extended/code/PythonProject/DeepingLearning/sofasofa/prac_4/data/test.csv'

    def loading(self):
        data_train = pd.read_csv(self.file_train)
        del data_train['CaseId']
        data_test = pd.read_csv(self.file_test)
        del data_test['CaseId']
        return data_train.values, data_test.values

    def final_train_test(self):
        data_train, data_test = self.loading()
        train_x, train_y = data_train[:, 0: -1], data_train[:, -1]
        return train_x, train_y, data_test, None

    def train_test_data(self):
        """[summary]
        
        Arguments:
            stand {[type]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        data_train, data_test = self.loading()
        __myscale = scale.MinMaxScaler(feature_range=(0, 1))
        train_x, test_x, train_y, test_y = train_test_split(data_train[:, 0: -1], data_train[:, -1],
                                                            test_size=0.2,
                                                            shuffle=True)
        return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    test = LoadData()
    test.train_test_data()
