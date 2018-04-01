from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
import load_data as ld
import numpy as np
import math


class myLasso(object):
    def __init__(self, train_x, train_y, alpha=0.01):
        self.train_x = train_x
        self.train_y = train_y
        self._lasso = linear_model.Lasso(alpha=alpha, normalize=True)

    def train_lasso(self):
        self._lasso.fit(X=self.train_x, y=self.train_y)

    def pre_lasso(self, test_x):
        return self._lasso.predict(test_x)


class knnregress(object):
    def __init__(self):
        self.knn = KNeighborsRegressor()

    def train_knn(self, train_x, train_y):
        self.knn.fit(train_x, train_y)

    def pre_knn(self, test_x):
        return self.knn.predict(test_x)


class Mape(object):
    def __init__(self):
        pass

    def mape_error(self, y_pre, y_true):
        temp = 0.0
        length = len(y_true)
        for i in range(length):
            temp += math.fabs(y_pre[i] - y_true[i]) / y_true[i]
        return temp / length

    def paint_pic(self, y_true, y_pre):
        import matplotlib.pyplot as plt

        plt.plot(y_true, label='y_true')
        plt.plot(y_pre, label='y_pre')
        plt.legend()
        plt.show()

    def alpha(self, mape_error):
        return (1.0 / 4) * math.log((1 - mape_error) / mape_error, math.e)


if __name__ == '__main__':
    _mape = Mape()

    data = ld.load_data_train('lasso')[:, 1:]
    data_x = []
    for d in data[:, 0]:
        temp = map(int, d.split('-'))
        data_x.append(temp)
    data_x = np.array(data_x)
    data_y = data[:, 1:]
    train_size = int(len(data_x) * 0.7)
    train_x, test_x = data_x[0: train_size, :], data_x[train_size:, :]
    train_y, test_y = data_y[: train_size, :], data_y[train_size:, :]
    lasso = myLasso(train_x=train_x, train_y=train_y[:, 0], alpha=0.5)
    lasso.train_lasso()
    lasso_pre_y = lasso.pre_lasso(test_x=test_x)
    lass_mape = _mape.mape_error(y_true=test_y[:, 0], y_pre=lasso_pre_y)
    print(lass_mape)

    q_error = [test_y[i, 0] - lasso_pre_y[i] for i in range(len(lasso_pre_y))]
    # a_error = [test_y[i] - lasso_pre_y[i] for i in range(lasso_pre_y)]

    knn = knnregress()
    knn.train_knn(train_x=train_x, train_y=train_y[:, 0])
    knn_pre_y = knn.pre_knn(test_x=test_x)
    knn_mape = _mape.mape_error(y_true=test_y[:, 0], y_pre=knn_pre_y)
    print(knn_mape)

    final_pre = (lasso_pre_y * _mape.alpha(lass_mape) + knn_pre_y * _mape.alpha(knn_mape))

    print(_mape.mape_error(y_true=test_y[:, 0], y_pre=final_pre))
    _mape.paint_pic(y_true=test_y, y_pre=final_pre)
