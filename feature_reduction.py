from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR


class FeatureReduction:

    def __init__(self, task, X, y=None):
        """
        Constructor
        :param task: 'reg' for regression; 'bin' for binary classification; 'multi' for multiclass classification
        :param X: DataFrame, with all features (do not encode categorical features; drop IDs or high cardinality features)
        :param y: Series, the target
        """
        self.X = X
        self.y = y
        self.task = task

    # test passed
    def tree_based(self, n_keep, n_trees, depth):
        """
        Use a Random Forest model to find the most important features
        :param n_keep: number of features to keep
        :param n_trees: number of trees in Random Forest
        :param depth: tree depth in Random Forest
        :param encoded:
                True: DataFrames are already encoded
                False: DataFrames are not encoded, will be encoded by LabelEncoder and StandardScaler
        :return
                list of kept features
                DataFrame with only kept features
        """
        if self.task == 'reg':
            model = RFR(n_estimators=n_trees, max_depth=depth, n_jobs=-1)
        else:
            model = RFC(n_estimators=n_trees, max_depth=depth, n_jobs=-1)
        model.fit(self.X, self.y)
        ft_imp = sorted(zip(self.X.columns.to_list(), model.feature_importances_), reverse=True)
        top_list = [feature for feature, score in ft_imp]
        return top_list[:n_keep], self.X[top_list[:n_keep]]

    def lasso(self, n_keep, l1=None):
        if l1 is None:
            l1 = [0.1, 1, 10]
