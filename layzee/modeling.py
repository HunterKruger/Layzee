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

    def __init__(self, X_train, X_test, y_train, y_test, task, metric, parallelism=-1, cv=3, random_state=1337):
        """
        Initialize data and hyper-parameter settings.
        :param X_train: training set features, DataFrame
        :param y_train: training set target, Series
        :param X_test: test set features, DataFrame
        :param y_test: test set target, Series
        :param task:
            'bin' for binary classification
            'mlt' for multi-class classification
            'reg' for regression
        :param parallelism: number of cores used for parallel training, -1 to use all
        :param metric: optimize model hyper-parameters for this metric during validation process
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
        self.parallelism = parallelism
        self.random_state = random_state
        self.cv = cv

    def hyperparam_mapping(self, model):

        reg_model_hp = {
            'xgb': {'n_estimators': [50, 100, 300],  # Number of gradient boosted trees.
                    'eta': [0.1, 0.2, 0.3],  # Learning rate.
                    'max_depth': [3, 7, 10],  # Maximum tree depth for base learners.
                    'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight(hessian) needed in a child.
                    'subsample': [1, 0.5],  # Subsample ratio of the training instance.
                    'colsample_bytree': [1, 0.5],  # Subsample ratio of columns when constructing each tree.
                    'lambda': [1, 10, 0.1],  # L2 regularization term on weights
                    'alpha': [1, 10, 0.1],  # L1 regularization term on weights
                    'Gamma': [0, 1],
                    # Minimum loss reduction required to make a further partition on a leaf node of the tree.
                    },
            'rf': {'n_estimators': [50, 100, 300],  # The number of trees in the forest.
                   'max_depth': [3, 7, 10],  # The maximum depth of the tree.
                   'min_samples_leaf': [1, 3, 5]  # The minimum number of samples required to be at a leaf node.
                   },
            'knn': {
                'n_neighbors': [3, 5, 9],  # Number of neighbors to use by default for kneighbors queries.
                'weights': ['distance'],  # weight function used in prediction
                'p': [1, 2]  # Power parameter for the Minkowski metric.
            },
            'lr': {},
            'lasso': {
                'alpha': [0.1, 1, 10]  # Regularization strength; must be a positive float.
            },
            'ridge': {
                'alpha': [0.1, 1, 10]  # Regularization strength; must be a positive float.
            },
            'dt': {
                'max_depth': [5, 10, 20],  # The maximum depth of the tree.
                'min_samples_leaf': [1, 5, 10]  # The minimum number of samples required to be at a leaf node.
            },
            'svm': {
                'kernel': ['rbf', 'linear'],  # SVM kernel
                'gamma': ['scale', 'auto'],
                # Gamma defines the 'influence' of each training example in the features space.
                'C': [0.1, 1, 10],
                # Regularization parameter. The strength of the regularization is inversely proportional to C.
            },
            'gb': {
                'loss': ['ls', 'lad', 'huber'],
                # Loss function to be optimized.
                # ‘ls’ refers to least squares regression.
                # ‘lad’ (least absolute deviation) is a highly robust
                #  loss function solely based on order information of the input variables.
                # ‘huber’ is a combination of the two.
                'learning_rate': [0.1, 0.2, 0.3],
                # Learning rate shrinks the contribution of each tree by learning_rate
                'n_estimators': [50, 100, 300],  # Number of gradient boosted trees.
                'max_depth': [3, 7, 10],  # The maximum depth of the tree.
            },
            'et': {'n_estimators': [50, 100, 300],  # The number of trees in the forest.
                   'max_depth': [3, 7, 10],  # The maximum depth of the tree.
                   'min_samples_leaf': [1, 3, 5]  # The minimum number of samples required to be at a leaf node.
                   },
            'sgd': {'loss': ['huber', 'squared_loss'],
                    # Selecting 'squared' loss will make the SGD behave like a Linear (OLS, LASSO and Ridge) Regression.
                    # Enabling 'Huber' loss will make the SGD more robust to outliers.
                    'max_iter': [100],  # Maximum number of iterations on the train data
                    'tol': [0.001],  # Tolerance for stopping criterion
                    'epsilon': [0.1, 1],  # Epsilon in the epsilon-insensitive loss functions
                    'alpha': [0.1, 1, 10],  # Regularization parameter.
                    'penalty': ['l2', 'l1', 'elasticnet']  # The penalty (aka regularization term) to be used.
                    },
            'lgb': {
                'learning_rate': [0.1, 0.3, 1],  # Boosting learning rate
                'n_estimators': [50, 100, 300],  # Number of boosted trees to fit
                'max_depth': [5, 9, 17],  # Maximum tree depth for base learners
                'subsample': [0.5, 1],  # Subsample ratio of the training instance.
                'colsample_bytree': [0.5, 1],  # Subsample ratio of columns when constructing each tree.
                'reg_alpha': [0.1, 1, 10],  # L1 regularization term on weights
                'reg_lambda': [0.1, 1, 10],  # L2 regularization term on weights
                'min_child_samples': [20, 50, 100],  # Minimum number of data needed in a child (leaf).
            },
            'cat': {
                'learning_rate': [0.1, 0.3, 1],  # Boosting learning rate
                'max_depth': [5, 9, 17],  # Maximum tree depth for base learners
                'n_estimators': [50, 100, 300],  # Number of boosted trees to fit
                'subsample': [0.5, 1],  # Subsample ratio of the training instance.
                'min_child_samples': [20, 50, 100],  # Minimum number of data needed in a child (leaf).
                'reg_lambda': [0.1, 1, 10]  # L2 regularization term on weights
            }
        }

        bin_model_hp = {
            'xgb': {'n_estimators': [50, 100, 300],  # Number of gradient boosted trees.
                    'eta': [0.1, 0.2, 0.3],  # Learning rate.
                    'max_depth': [3, 7, 10],  # Maximum tree depth for base learners.
                    'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight(hessian) needed in a child.
                    'subsample': [1, 0.5],  # Subsample ratio of the training instance.
                    'colsample_bytree': [1, 0.5],  # Subsample ratio of columns when constructing each tree.
                    'lambda': [1, 10, 0.1],  # L2 regularization term on weights
                    'alpha': [1, 10, 0.1],  # L1 regularization term on weights
                    'Gamma': [0, 1],
                    # Minimum loss reduction required to make a further partition on a leaf node of the tree.
                    'class_weight': ['balanced']
                    },
            'rf': {'n_estimators': [50, 100, 300],  # The number of trees in the forest.
                   'max_depth': [3, 7, 10],  # The maximum depth of the tree.
                   'min_samples_leaf': [1, 3, 5],  # The minimum number of samples required to be at a leaf node.
                   'class_weight': ['balanced']
                   },
            'knn': {
                'n_neighbors': [3, 5, 9],  # Number of neighbors to use by default for kneighbors queries.
                'weights': ['distance'],  # weight function used in prediction
                'p': [1, 2]  # Power parameter for the Minkowski metric.
            },
            'lr': {
                'penalty': ['l1', 'l2', 'elasticnet'],  # Used to specify the norm used in the penalization.
                'C': [0.1, 0.01, 1, 10, 100],  # Inverse of regularization strength
                'class_weight': ['balanced']
            },
            'dt': {
                'max_depth': [5, 10, 20],  # The maximum depth of the tree.
                'min_samples_leaf': [1, 5, 10],  # The minimum number of samples required to be at a leaf node.
                'class_weight': ['balanced']
            },
            'svm': {
                'kernel': ['rbf', 'linear'],  # SVM kernel
                'gamma': ['scale', 'auto'],
                # Gamma defines the 'influence' of each training example in the features space.
                'C': [0.1, 1, 10],
                # Regularization parameter. The strength of the regularization is inversely proportional to C.
                'class_weight': ['balanced']
            },
            'gb': {
                'loss': ['deviance', 'exponential'],
                # Deviance refers to deviance (= logistic regression) for classification with probabilistic outputs.
                # For loss 'exponential', gradient boosting recovers the AdaBoost algorithm.
                'learning_rate': [0.1, 0.2, 0.3],
                # Learning rate shrinks the contribution of each tree by learning_rate
                'n_estimators': [50, 100, 300],  # Number of gradient boosted trees.
                'max_depth': [3, 7, 10],  # The maximum depth of the tree.
            },
            'et': {'n_estimators': [50, 100, 300],  # The number of trees in the forest.
                   'max_depth': [3, 7, 10],  # The maximum depth of the tree.
                   'min_samples_leaf': [1, 3, 5],  # The minimum number of samples required to be at a leaf node.
                   'class_weight': ['balanced']
                   },
            'sgd': {'loss': ['huber', 'squared_loss'],
                    # Selecting 'squared' loss will make the SGD behave like a Linear (OLS, LASSO and Ridge) Regression.
                    # Enabling 'Huber' loss will make the SGD more robust to outliers.
                    'max_iter': [100],  # Maximum number of iterations on the train data
                    'tol': [0.001],  # Tolerance for stopping criterion
                    'epsilon': [0.1, 1],  # Epsilon in the epsilon-insensitive loss functions
                    'alpha': [0.1, 1, 10],  # Regularization parameter.
                    'penalty': ['l2', 'l1', 'elasticnet'],  # The penalty (aka regularization term) to be used.
                    'class_weight': ['balanced']
                    },
            'lgb': {
                'learning_rate': [0.1, 0.3, 1],  # Boosting learning rate
                'n_estimators': [50, 100, 300],  # Number of boosted trees to fit
                'max_depth': [5, 9, 17],  # Maximum tree depth for base learners
                'subsample': [0.5, 1],  # Subsample ratio of the training instance.
                'colsample_bytree': [0.5, 1],  # Subsample ratio of columns when constructing each tree.
                'reg_alpha': [0.1, 1, 10],  # L1 regularization term on weights
                'reg_lambda': [0.1, 1, 10],  # L2 regularization term on weights
                'min_child_samples': [20, 50, 100],  # Minimum number of data needed in a child (leaf).
                'class_weight': ['balanced']
            },
            'cat': {
                'learning_rate': [0.1, 0.3, 1],  # Boosting learning rate
                'max_depth': [5, 9, 17],  # Maximum tree depth for base learners
                'n_estimators': [50, 100, 300],  # Number of boosted trees to fit
                'subsample': [0.5, 1],  # Subsample ratio of the training instance.
                'min_child_samples': [20, 50, 100],  # Minimum number of data needed in a child (leaf).
                'reg_lambda': [0.1, 1, 10],  # L2 regularization term on weights
                'auto_class_weights': ['balanced']
            }
        }

        mlt_model_hp = {
            'xgb': {'n_estimators': [50, 100, 300],  # Number of gradient boosted trees.
                    'eta': [0.1, 0.2, 0.3],  # Learning rate.
                    'max_depth': [3, 7, 10],  # Maximum tree depth for base learners.
                    'min_child_weight': [1, 3, 5],  # Minimum sum of instance weight(hessian) needed in a child.
                    'subsample': [1, 0.5],  # Subsample ratio of the training instance.
                    'colsample_bytree': [1, 0.5],  # Subsample ratio of columns when constructing each tree.
                    'lambda': [1, 10, 0.1],  # L2 regularization term on weights
                    'alpha': [1, 10, 0.1],  # L1 regularization term on weights
                    'Gamma': [0, 1],
                    # Minimum loss reduction required to make a further partition on a leaf node of the tree.
                    'class_weight': ['balanced']
                    },
            'rf': {'n_estimators': [50, 100, 300],  # The number of trees in the forest.
                   'max_depth': [3, 7, 10],  # The maximum depth of the tree.
                   'min_samples_leaf': [1, 3, 5],  # The minimum number of samples required to be at a leaf node.
                   'class_weight': ['balanced']
                   },
            'knn': {
                'n_neighbors': [3, 5, 9],  # Number of neighbors to use by default for kneighbors queries.
                'weights': ['distance'],  # weight function used in prediction
                'p': [1, 2]  # Power parameter for the Minkowski metric.
            },
            'lr': {
                'penalty': ['l1', 'l2', 'elasticnet'],  # Used to specify the norm used in the penalization.
                'C': [0.1, 0.01, 1, 10, 100],  # Inverse of regularization strength
                'class_weight': ['balanced']
            },
            'dt': {
                'max_depth': [5, 10, 20],  # The maximum depth of the tree.
                'min_samples_leaf': [1, 5, 10],  # The minimum number of samples required to be at a leaf node.
                'class_weight': ['balanced']
            },
            'svm': {
                'kernel': ['rbf', 'linear'],  # SVM kernel
                'gamma': ['scale', 'auto'],
                # Gamma defines the 'influence' of each training example in the features space.
                'C': [0.1, 1, 10],
                # Regularization parameter. The strength of the regularization is inversely proportional to C.
                'class_weight': ['balanced']
            },
            'gb': {
                'loss': ['deviance', 'exponential'],
                # Deviance refers to deviance (= logistic regression) for classification with probabilistic outputs.
                # For loss 'exponential', gradient boosting recovers the AdaBoost algorithm.
                'learning_rate': [0.1, 0.2, 0.3],
                # Learning rate shrinks the contribution of each tree by learning_rate
                'n_estimators': [50, 100, 300],  # Number of gradient boosted trees.
                'max_depth': [3, 7, 10],  # The maximum depth of the tree.
            },
            'et': {'n_estimators': [50, 100, 300],  # The number of trees in the forest.
                   'max_depth': [3, 7, 10],  # The maximum depth of the tree.
                   'min_samples_leaf': [1, 3, 5],  # The minimum number of samples required to be at a leaf node.
                   'class_weight': ['balanced']
                   },
            'sgd': {'loss': ['huber', 'squared_loss'],
                    # Selecting 'squared' loss will make the SGD behave like a Linear (OLS, LASSO and Ridge) Regression.
                    # Enabling 'Huber' loss will make the SGD more robust to outliers.
                    'max_iter': [100],  # Maximum number of iterations on the train data
                    'tol': [0.001],  # Tolerance for stopping criterion
                    'epsilon': [0.1, 1],  # Epsilon in the epsilon-insensitive loss functions
                    'alpha': [0.1, 1, 10],  # Regularization parameter.
                    'penalty': ['l2', 'l1', 'elasticnet'],  # The penalty (aka regularization term) to be used.
                    'class_weight': ['balanced']
                    },
            'lgb': {
                'learning_rate': [0.1, 0.3, 1],  # Boosting learning rate
                'n_estimators': [50, 100, 300],  # Number of boosted trees to fit
                'max_depth': [5, 9, 17],  # Maximum tree depth for base learners
                'subsample': [0.5, 1],  # Subsample ratio of the training instance.
                'colsample_bytree': [0.5, 1],  # Subsample ratio of columns when constructing each tree.
                'reg_alpha': [0.1, 1, 10],  # L1 regularization term on weights
                'reg_lambda': [0.1, 1, 10],  # L2 regularization term on weights
                'min_child_samples': [20, 50, 100],  # Minimum number of data needed in a child (leaf).
                'class_weight': ['balanced']
            },
            'cat': {
                'learning_rate': [0.1, 0.3, 1],  # Boosting learning rate
                'max_depth': [5, 9, 17],  # Maximum tree depth for base learners
                'n_estimators': [50, 100, 300],  # Number of boosted trees to fit
                'subsample': [0.5, 1],  # Subsample ratio of the training instance.
                'min_child_samples': [20, 50, 100],  # Minimum number of data needed in a child (leaf).
                'reg_lambda': [0.1, 1, 10],  # L2 regularization term on weights
                'auto_class_weights': ['balanced']
            }

        }

        if self.task == 'reg':
            return reg_model_hp[model]
        if self.task == 'bin':
            return bin_model_hp[model]
        if self.task == 'mlt':
            return mlt_model_hp[model]

    def model_mapping(self, model):
        bin_model_mapper = {
            'svm': SVC(probability=True, random_state=self.random_state),
            'rf': RandomForestClassifier(random_state=self.random_state, n_jobs=self.parallelism),
            'gb': GradientBoostingClassifier(random_state=self.random_state),
            'et': ExtraTreesClassifier(random_state=self.random_state, n_jobs=self.parallelism),
            'dt': DecisionTreeClassifier(random_state=self.random_state),
            'lr': LogisticRegression(random_state=self.random_state),
            'knn': KNeighborsClassifier(),
            'sgd': SGDClassifier(random_state=self.random_state),
            'xgb': XGBClassifier(objective='binary:logistic', random_state=self.random_state, n_jobs=self.parallelism),
            'lgb': LGBMClassifier(objective='binary', random_state=self.random_state, n_jobs=self.parallelism),
            'cat': CatBoostClassifier(objective='Logloss', random_state=self.random_state,
                                      thread_count=self.parallelism)
        }

        mlt_model_mapper = {
            'svm': SVC(probability=True, random_state=self.random_state),
            'rf': RandomForestClassifier(random_state=self.random_state, n_jobs=self.parallelism),
            'gb': GradientBoostingClassifier(random_state=self.random_state),
            'et': ExtraTreesClassifier(random_state=self.random_state, n_jobs=self.parallelism),
            'dt': DecisionTreeClassifier(random_state=self.random_state),
            'lr': LogisticRegression(random_state=self.random_state),
            'knn': KNeighborsClassifier(),
            'sgd': SGDClassifier(random_state=self.random_state),
            'xgb': XGBClassifier(objective='multi:softprob', random_state=self.random_state, n_jobs=self.parallelism),
            'lgb': LGBMClassifier(objective='multiclass', random_state=self.random_state, n_jobs=self.parallelism),
            'cat': CatBoostClassifier(objective='MultiClass', random_state=self.random_state,
                                      thread_count=self.parallelism)
        }

        reg_model_mapper = {
            'svm': SVR(),
            'rf': RandomForestRegressor(random_state=self.random_state, n_jobs=self.parallelism),
            'gb': GradientBoostingRegressor(random_state=self.random_state),
            'et': ExtraTreesRegressor(random_state=self.random_state, n_jobs=self.parallelism),
            'dt': DecisionTreeRegressor(random_state=self.random_state),
            'knn': KNeighborsRegressor(),
            'sgd': SGDRegressor(random_state=self.random_state),
            'xgb': XGBRegressor(objective='reg:squarederror', random_state=self.random_state, n_jobs=self.parallelism),
            'lasso': Lasso(random_state=self.random_state),
            'ridge': Ridge(random_state=self.random_state),
            'lr': LinearRegression(),
            'lgb': LGBMRegressor(objective='regression', random_state=self.random_state, n_jobs=self.parallelism),
            'cat': CatBoostRegressor(objective='RMSE', random_state=self.random_state, thread_count=self.parallelism)
        }

        if self.task == 'reg':
            return reg_model_mapper[model]
        if self.task == 'bin':
            return bin_model_mapper[model]
        if self.task == 'mlt':
            return mlt_model_mapper[model]

    def modeling(self, model, hp='auto', strategy='grid', max_iter=10, n_jobs=-1, calibration=None):
        """
        Modeling with a model and its specified hyper-parameters.
        :param model: a model name in short, check 'model_mapping' function
        :param hp: a dictionary, key: hyper-parameter, value: list or range of hyper-parameter
                   use pre-set hyper-parameter space if 'auto'
        :param strategy:
             None for fitting directly
            'grid' for Grid search
            'random' for Random search
            'bayesian' for Bayesian Search, currently not supported
        :param max_iter: max iteration for random search
        :param n_jobs: number of hardware sources for hyper-params searching, -1 to use all
        :param calibration: calibrate model output to more accurate probability,
                            not available for regression tasks,
                            no need for LogisticRegression or SVC models
            None: no calibration
            'sigmoid': platt scaling
            'isotonic': isotonic regression
        :return
            y_score: prediction of the best model in probability, for positive label, 1d array-like
            y_proba: prediction of the best model in probability, for each label (only for multiclass task)
            best_model: best model object
            best_score: best score in the specified metric
        """
        start = datetime.datetime.now()

        if hp == 'auto':
            hp = self.hyperparam_mapping(model)
        internal_model = self.model_mapping(model)
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
            if calibration is None:
                y_score = best_model.predict(self.X_test)
            else:
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
