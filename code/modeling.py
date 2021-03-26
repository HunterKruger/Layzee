import datetime

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor, Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, GradientBoostingRegressor, \
    ExtraTreesRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.calibration import CalibratedClassifierCV


class Modeling:

    def __init__(self, X_train, X_test, y_train, y_test, task, metric, cv=3, random_state=1337):
        """
        Constructor.
        Initialize data and hyper-parameter settings.
        :param X_train: training set features, DataFrame
        :param y_train: training set target, Series
        :param X_test: test set features, DataFrame
        :param y_test: test set target, Series
        :param task:
            'bin' for binary classification
            'mlt' for multi-class classification
            'reg' for regression
        :param metric: optimize model hyper-parameters for this metric
            see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        :param cv: if > 0 and < 1, simple train/validation split, indicate validation set ratio,
                   if > 1 and int, cross validation, indicate folds
        :param random_state: random state seed
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        self.task = task
        self.metric = metric
        self.random_state = random_state
        self.cv = cv

    def model_mapping(self, model, class_weight):
        bin_model_mapper = {
            'svm': SVC(probability=True, class_weight=class_weight, random_state=self.random_state),
            'rf': RandomForestClassifier(random_state=self.random_state, class_weight=class_weight),
            'gb': GradientBoostingClassifier(random_state=self.random_state),
            'et': ExtraTreesClassifier(random_state=self.random_state, class_weight=class_weight),
            'dt': DecisionTreeClassifier(random_state=self.random_state, class_weight=class_weight),
            'lr': LogisticRegression(random_state=self.random_state, class_weight=class_weight),
            'knn': KNeighborsClassifier(),  # how to set class_weight? No random_state?
            'sgd': SGDClassifier(random_state=self.random_state, class_weight=class_weight),
            'xgb': XGBClassifier(objective='binary:logistic', random_state=self.random_state),
            'lgb': LGBMClassifier(class_weight=class_weight, random_state=self.random_state),
            'cat': CatBoostClassifier(random_state=self.random_state, auto_class_weights=class_weight)
        }

        mlt_model_mapper = {
            'svm': SVC(probability=True, class_weight=class_weight, random_state=self.random_state),
            'rf': RandomForestClassifier(random_state=self.random_state, class_weight=class_weight),
            'gb': GradientBoostingClassifier(random_state=self.random_state),
            'et': ExtraTreesClassifier(random_state=self.random_state, class_weight=class_weight),
            'dt': DecisionTreeClassifier(random_state=self.random_state, class_weight=class_weight),
            'lr': LogisticRegression(random_state=self.random_state, class_weight=class_weight),
            'knn': KNeighborsClassifier(),
            'sgd': SGDClassifier(random_state=self.random_state, class_weight=class_weight),
            'xgb': XGBClassifier(objective='multi:softprob', random_state=self.random_state),
            'lgb': LGBMClassifier(class_weight=class_weight, random_state=self.random_state),
            'cat': CatBoostClassifier(random_state=self.random_state, auto_class_weights=class_weight)
        }

        reg_model_mapper = {
            'svm': SVR(),
            'rf': RandomForestRegressor(random_state=self.random_state),
            'gb': GradientBoostingRegressor(random_state=self.random_state),
            'et': ExtraTreesRegressor(random_state=self.random_state),
            'dt': DecisionTreeRegressor(random_state=self.random_state),
            'knn': KNeighborsRegressor(),
            'sgd': SGDRegressor(random_state=self.random_state),
            'xgb': XGBRegressor(objective='reg:squarederror', random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'ridge': Ridge(random_state=self.random_state),
            'lr': LinearRegression(),
            'lgb': LGBMRegressor(random_state=self.random_state),
            'cat': CatBoostRegressor(random_state=self.random_state)
        }
        if self.task == 'reg':
            return reg_model_mapper[model]
        if self.task == 'bin':
            return bin_model_mapper[model]
        if self.task == 'mlt':
            return mlt_model_mapper[model]

    def modeling(self, model, hp, strategy='grid', max_iter=10, n_jobs=-1, class_weight=None, calibration=None):
        """
        Modeling with a model and its specified hyper-parameters.
        :param model: a model name
        :param hp: a dictionary, key: hyper-param name, value: list or range of hyper-param
        :param strategy:
            'grid' for Grid search
            'random' for Random search
            'bayesian' for Bayesian Search, currently not supported
        :param max_iter: max iteration for random search
        :param n_jobs: number of hardware sources for hyper-params searching, -1 to use all
        :param class_weight: only for classification tasks
            None: all rows will be considered equally.
            'balanced': The "balanced" mode uses the values of y to automatically adjust
                        weights inversely proportional to class frequencies in the input data
                        as ``n_samples / (n_classes * np.bincount(y))``
        :param calibration: calibrate model output to more accurate probability,
                            not available for regression tasks,
                            no need for LogisticRegression or SVC models
            None: no calibration
            'sigmoid': platt scaling
            'isotonic': isotonic regression
        :return
            y_score: prediction of the best model, 1d array-like
            y_proba: prediction of the best model in probability, for each label (only for multiclass task)
            best_model: best model object
            best_score: best score in the specified metric
        """
        start = datetime.datetime.now()

        internal_model = self.model_mapping(model, class_weight)
        optimizer = None
        if strategy == 'grid' and 0 < self.cv < 1:  # grid, hold-out
            optimizer = GridSearchCV(estimator=internal_model,
                                     param_grid=hp,
                                     n_jobs=n_jobs,
                                     scoring=self.metric,
                                     refit=True,
                                     cv=ShuffleSplit(test_size=self.cv,
                                                     n_splits=1,
                                                     random_state=self.random_state)
                                     )
        if strategy == 'grid' and self.cv >= 1:  # grid, cv
            optimizer = GridSearchCV(estimator=internal_model,
                                     param_grid=hp,
                                     n_jobs=n_jobs,
                                     cv=self.cv,
                                     scoring=self.metric,
                                     refit=True
                                     )
        if strategy == 'random' and 0 < self.cv < 1:  # random, hold-out
            optimizer = RandomizedSearchCV(estimator=internal_model,
                                           param_distributions=hp,
                                           n_iter=max_iter,
                                           n_jobs=n_jobs,
                                           refit=True,
                                           scoring=self.metric,
                                           cv=ShuffleSplit(test_size=self.cv,
                                                           n_splits=1,
                                                           random_state=self.random_state)
                                           )
        if strategy == 'random' and self.cv >= 1:  # random, cv
            optimizer = RandomizedSearchCV(estimator=internal_model,
                                           param_distributions=hp,
                                           n_iter=max_iter,
                                           n_jobs=n_jobs,
                                           cv=self.cv,
                                           scoring=self.metric,
                                           refit=True
                                           )

        optimizer.fit(self.X_train, self.y_train)
        best_model = optimizer.best_estimator_
        best_score = optimizer.best_score_
        best_params = optimizer.best_params_

        # calibration:
        calib_clf = None
        if calibration == 'sigmoid':
            calib_clf = CalibratedClassifierCV(base_estimator=best_model, method='sigmoid', cv='prefit')
            calib_clf.fit(self.X_test, self.y_test)
        if calibration == 'isotonic':
            calib_clf = CalibratedClassifierCV(base_estimator=best_model, method='isotonic', cv='prefit')
            calib_clf.fit(self.X_test, self.y_test)

        end = datetime.datetime.now()
        print(end - start)

        if self.task == 'reg':
            y_score = calib_clf.predict(self.X_test)
            return y_score, best_model, best_score, best_params
        if self.task == 'bin':
            if calibration is None:
                y_score = best_model.predict_proba(self.X_test)
            else:
                y_score = calib_clf.predict_proba(self.X_test)
            return y_score[:, 1], best_model, best_score, best_params
        if self.task == 'mlt':
            y_score = best_model.predict(self.X_test)
            if calibration is None:
                y_proba = best_model.predict_proba(self.X_test)
            else:
                y_proba = calib_clf.predict_proba(self.X_test)
            return y_score, y_proba, best_model, best_score, best_params
