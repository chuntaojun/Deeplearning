from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score


class RandomForest_Model(object):
    """[summary]
    
    Arguments:
        object {[type]} -- [description]
    """

    def __init__(self, n_estimators, data_x, data_y):
        super(RandomForest_Model, self).__init__()
        self.n_estimators = n_estimators
        self.data_x = data_x
        self.data_y = data_y
    
    def build_model(self):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=0)
        print "build randomforest model success"

    def train_model(self):
        if self.model is None:
            raise Exception("Please run obj.build_model() function")
        print "start train model..."
        self.model.fit(self.data_x, self.data_y)
        print "train model had finished"
    
    def prediction(self, test_x):
        return self.model.predict_proba(test_x)[:, 1]


from sklearn.ensemble import AdaBoostClassifier


class AdaBoost_Model(object):
    def __init__(self, n_estimators, data_x, data_y):
        super(AdaBoost_Model, self).__init__
        self.n_estimators = n_estimators
        self.data_x = data_x
        self.data_y = data_y
    
    def build_model(self):
        self.model = AdaBoostClassifier(n_estimators=self.n_estimators)
    
    def train_model(self):
        if self.model is None:
            raise Exception("you hava build model before train_model() function")
        self.model.fit(self.data_x, self.data_y)
    
    def prediction(self, test_x):
        self.model.predict_proba(test_x)[:, 1]


if __name__ == '__main__':
    import load_file
    import write_file

    LoadData = load_file.LoadData()

    train_x, train_y, test_x, test_y = LoadData.train_test_data()
    _model = RandomForest_Model(n_estimators=100, data_x=train_x, data_y=train_y)
    _model.build_model()
    _model.train_model()
    pre = _model.prediction(test_x)
    print average_precision_score(y_true=test_y, y_score=pre)
