from pandas import DataFrame, Series


class Modeling:

    def __init__(self, X_train: DataFrame, y_train: Series, X_test: DataFrame, y_test: Series, task):
        """
        Constructor
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param task: 'bin_cls': binary classification,
                     'multi_cls', multi-class classification
                     'reg': regression
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.task = task

    def find_best_param_cv(self, model, param, fold=3):
        pass
