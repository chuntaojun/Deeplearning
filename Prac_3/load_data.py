import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


FILE_TRAIN_PATH = '/Volumes/Extended/code/PythonProject/DeepingLearning/sofasofa/Prac_3/data/train.csv'
FILE_TEST_PATH = '/Volumes/Extended/code/PythonProject/DeepingLearning/sofasofa/Prac_3/data/test.csv'
FILL_NAME = [[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]]


def str2bin(user_name):
    """

    :param user_name:
    :return:
    """
    return [map(int, bin(ord(i))[2:]) for i in user_name[0]]


def listExtend(List):
    """

    :param List:
    :return:
    """
    a = []
    if len(List) == 3:
        List.extend(FILL_NAME)
        [a.extend(i) for i in List]
    else:
        [a.extend(i) for i in List]
    return a


def load_data_dense(fileName_train=FILE_TRAIN_PATH, fileName_test=FILE_TEST_PATH):
    """

    :param fileName_train:
    :param fileName_test:
    :return:
    """
    train, test = np.array(pd.read_csv(fileName_train).values), np.array(pd.read_csv(fileName_test).values)
    train_sex = train[:, -1]
    train_name, test_name = map(str2bin, train[:, 1:2]), map(str2bin, test[:, 1:2])

    def name2vec(names):
        return np.array(map(listExtend, names))

    train_name = name2vec(train_name);test_name = name2vec(test_name)
    train_x, test_x, train_y, test_y = train_test_split(train_name, train_sex, test_size=0.3)
    return train_x, train_y, test_x, test_y


def load_data_conv(fileName_train=FILE_TRAIN_PATH, fileName_test=FILE_TEST_PATH):
    """

    :param fileName_train:
    :param fileName_test:
    :return:
    """
    train, test = np.array(pd.read_csv(fileName_train).values), np.array(pd.read_csv(fileName_test).values)
    sex = train[:, -1]
    import jieba
    temp_name = []
    for name in train:
        temp_name.append(' '.join(jieba.cut(name[0], cut_all=True)))
    tfidef_vectorizer = TfidfVectorizer()
    tfidf_maxtir = tfidef_vectorizer.fit_transform(temp_name)
    train_data = np.column_stack((np.array(tfidf_maxtir.todense()), sex))
    train_name, test_name, train_sex, test_sex = train_test_split(train_data[:, 1:2], train_data[:, 2], test_size=0.2)
    return train_name, train_sex.transpose(), test_name, test_sex


def load_data(type=1):
    """

    :param type:
    :return:
    """
    if type == 'dense':
        return load_data_dense()
    return load_data_conv()


if __name__ == '__main__':
    load_data('dense')
