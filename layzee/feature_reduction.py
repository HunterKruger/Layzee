from abc import abstractmethod
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, chi2, \
    f_classif, f_regression, SelectKBest, VarianceThreshold


class FeatureReduction:
    """
    Feature reduction operates on the preprocessed features.
    It allows you to reduce the dimension of the feature space
    in order to regularize your model or make it more interpretable.
    """

    def __init__(self, X, y=None):
        """
        Constructor
        :param X: DataFrame, with all features encoded
        :param y: Series, the target, encoded, optional
        """
        self.X = X
        self.y = y

    # test passed
    @abstractmethod
    def tree_based(self, n_keep, n_trees, depth):
        """
        This creates a Random Forest model to predict the target.
        Only the top features according to the feature importance computed by the algorithm will be selected.
        :param n_keep: number of features to keep
        :param n_trees: number of trees in Random Forest
        :param depth: tree depth in Random Forest
        :return
                list of top features,
                DataFrame with only top features
        """
        pass

    # test passed
    @abstractmethod
    def lasso(self, l1=None):
        pass

    @abstractmethod
    def mutual_info(self, n_keep):
        pass

    @abstractmethod
    def f(self, n_keep):
        pass

    def variance_based(self, threshold, num_cols='auto'):
        """
        Filter numerical features based on a specific threshold of variance
        :param threshold: threshold of std
        :param num_cols: list of str, numerical features
                         choose 'auto' for auto detection
        :return: reduced X
        """
        if num_cols == 'auto':
            num_cols = self.X.select_dtypes('number').columns.tolist()
        rest_cols = list(set(self.X.columns.tolist()) - set(num_cols))
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(self.X)
        return pd.concat([X_selected, self.X[rest_cols]], axis=1)

    def mode_based(self, threshold, cat_cols='auto'):
        """
        Filter categorical features based on a specific threshold of mode proportion
        :param threshold: threshold of mode proportion, 0-1.00 in decimal
        :param cat_cols: list of str, categorical features
                         choose 'auto' for auto detection
        :return: X reduced
        """
        if cat_cols == 'auto':
            cat_cols = self.X.select_dtypes('object').columns.tolist()
        X_temp = self.X.copy()
        for col in cat_cols:
            if self[col].value_counts(normalize=True).nlargest().values[0] > threshold:
                X_temp.drop(col, axis=1, inplace=True)
        return X_temp

    # test passed
    def pca(self, n):
        """
        The feature space dimension will be reduced using Principal Component Analysis.
        Only the top principal components will be selected.
        This method will generate non-interpretable feature names as its output.
        The model may be performant, but will not be interpretable.
        :param n:
            if int, number of components
            if decimal >0 and <1, select the number of components such that the amount of variance that needs to be
                explained is greater than the percentage specified by n_components
        :return
            reduced X
        """
        result = PCA(n_components=n).fit_transform(self.X)
        nb_cols = result.shape[1]
        cols = ['pc' + str(i) for i in range(nb_cols)]
        return pd.DataFrame(result, columns=cols)


class RegressionFeatureReduction(FeatureReduction):

    def __init__(self, X, y=None):
        """
        Constructor
        :param X: DataFrame, with all features encoded
        :param y: Series, the target, encoded
        """
        super().__init__(X, y)

    # test passed
    def tree_based(self, n_keep, n_trees, depth):
        """
        This creates a Random Forest model to predict the target.
        Only the top features according to the feature importance computed by the algorithm will be selected.
        :param n_keep: number of features to keep
        :param n_trees: number of trees in Random Forest
        :param depth: tree depth in Random Forest
        :return
                X with only top features
                list of top features
        """
        model = RandomForestRegressor(n_estimators=n_trees, max_depth=depth, n_jobs=-1)
        model.fit(self.X, self.y)
        ft_imp = sorted(zip(self.X.columns.to_list(), model.feature_importances_), reverse=True)
        top_list = [feature for feature, score in ft_imp]
        if n_keep > len(top_list):
            n_keep = len(top_list)
        return top_list[:n_keep], self.X[top_list[:n_keep]]

    # test passed
    def lasso(self, l1=None):
        """
        This creates a LASSO model to predict the target, using 3-fold cross-validation to select the best value
        of the regularization term. Only the features with nonzero coefficients will be selected.
        :param l1: list of floats, for l1 penalty
        :return:
                list of selected features,
                X with only selected features
        """
        if l1 is None:
            l1 = [0.10, 0.1, 1, 10, 100]
        model = LassoCV(cv=3, alphas=l1).fit(self.X, self.y)
        result_series = pd.Series(list(model.coef_), index=self.X.columns)
        result_features = result_series[result_series != 0].index.tolist()
        return result_features, self.X[result_features]

    def pearson_corr(self, n_keep, num_cols='auto'):
        """
        :param n_keep: number of numerical columns to be kept
        :param num_cols: list of numerical features in X to be calculated with y
        :return:
            list of selected features
            X with selected num_cols and non-num_cols together
        """
        if num_cols == 'auto':
            num_cols = self.X.select_dtypes('object').columns.tolist()
        if n_keep >= len(num_cols):
            raise ValueError("n must smaller than number of numerical cols!")
        else:
            X_temp = self.X[num_cols]
        rest_cols = list(set(self.X.columns.tolist()) - set(num_cols))
        all_data = pd.concat([X_temp, self.y], axis=1)
        selected_cols = all_data.corr('pearson').iloc[:-1, -1].sort_values(ascending=False).index.tolist()[:n_keep]
        return selected_cols, self.X[selected_cols + rest_cols]

    def mutual_info(self, n_keep):
        pass

    def f(self, n_keep):
        pass


class ClassificationFeatureReduction(FeatureReduction):

    def __init__(self, X, y=None):
        """
        Constructor
        :param X: DataFrame, with all features encoded
        :param y: Series, the target, encoded
        """
        super().__init__(X, y)

    # test passed
    def tree_based(self, n_keep, n_trees, depth):
        """
        This creates a Random Forest model to predict the target.
        Only the top features according to the feature importance computed by the algorithm will be selected.
        :param n_keep: number of features to keep
        :param n_trees: number of trees in Random Forest
        :param depth: tree depth in Random Forest
        :return
                list of top features,
                DataFrame with only top features
        """
        model = RandomForestClassifier(n_estimators=n_trees, max_depth=depth, n_jobs=-1)
        model.fit(self.X, self.y)
        ft_imp = sorted(zip(self.X.columns.to_list(), model.feature_importances_), reverse=True)
        top_list = [feature for feature, score in ft_imp]
        if n_keep > len(top_list):
            n_keep = len(top_list)
        return top_list[:n_keep], self.X[top_list[:n_keep]]

    # test passed
    def lasso(self, l1=None):
        """
        This creates a LASSO model to predict the target, using 3-fold cross-validation to select the best value
        of the regularization term. Only the features with nonzero coefficients will be selected.
        :param l1: list of floats, for l1 penalty
        :return:
                list of selected features,
                DataFrame with only selected features
        """
        if l1 is None:
            l1 = [0.10, 0.1, 1, 10, 100]
        model = LogisticRegressionCV(penalty='l1', cv=3, solver='liblinear', Cs=[1 / x for x in l1],
                                     class_weight='balanced', n_jobs=-1).fit(self.X, self.y)
        result_series = pd.Series(list(model.coef_[0]), index=self.X.columns)
        result_features = result_series[result_series != 0].index.tolist()
        return result_features, self.X[result_features]

    def mutual_info(self, n_keep):
        pass

    def chi2(self, n_keep):
        pass

    def f(self, n_keep):
        pass
