from pandas import DataFrame, Series
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureReduction:

    def __init__(self, task, X, y=None):
        """
        Constructor
        :param X: DataFrame, with all features (do not encode categorical features; drop IDs or high cardinality features)
        :param y: Series, the target
        :param task: 'reg' for regression; 'bin' for binary classification; 'multi' for multiclass classification
        """
        self.X = X
        self.y = y
        self.task = task

    def tree_based(self, n_keep, n_trees, depth, encoded=False):
        """
        Use a Random Forest model to find the most important features
        :param n_keep: number of features to keep
        :param n_trees: number of trees in Random Forest
        :param depth: tree depth in Random Forest
        :param encoded:
                True: DataFrames are already encoded
                False: DataFrames are not encoded, will be encoded by LabelEncoder and StandardScaler
        """
        if not encoded:
            X_cat = self.X.select_dtypes(include='object')
            X_num = self.X.select_dtypes(include='number')
            le = LabelEncoder()
            X_cat = le.fit_transform(X_cat)
            ss = StandardScaler()
            X_num = ss.fit_transform(X_num)
            X_new = pd.concat([X_num, X_cat], axis=1)
        else:
            X_new = self.df1
        if self.task == 'reg':
            model = RFR(n_estimators=n_trees, max_depth=depth, n_jobs=-1)
        else:
            model = RFC(n_estimators=n_trees, max_depth=depth, n_jobs=-1)
        model.fit(X_new, self.y)
        ft_imp = sorted(zip(X_new.columns.to_list(), model.feature_importances_), reverse=True)
        top_list = [feature for feature, score in ft_imp]
        return top_list[:n_keep]

    def lasso(self, n_keep, l1=None):
        if l1 is None:
            l1 = [0.1, 1, 10]
