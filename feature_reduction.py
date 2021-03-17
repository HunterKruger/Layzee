from abc import abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, chi2, f_classif, f_regression
import pandas as pd


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
        :param y: Series, the target, encoded
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

    def variance_based(self, threshold):
        pass

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
        Only the top features according to the feature importances computed by the algorithm will be selected.
        :param n_keep: number of features to keep
        :param n_trees: number of trees in Random Forest
        :param depth: tree depth in Random Forest
        :return
                list of top features,
                X with only top features
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

    # test passed
    def corr(self, n, cols, method='pearson', return_all=False):
        """
        :param method: 'pearson', 'spearman', 'kendall'
        :param n: number of numerical columns to be kept
        :param cols: list of columns in X to be calculated with y
        :param return_all: return selected num_cols and non-num_cols together
        :return:
        """

        if n >= len(cols):
            print("n must smaller than number of numerical cols!")
            return
        else:
            temp_X = self.X[cols]

        cols_rest = list(set(self.X.columns.tolist()) - set(temp_X.columns.tolist()))

        all_data = pd.concat([temp_X, self.y], axis=1)
        result = all_data.corr(method).iloc[:-1, -1].sort_values(ascending=False).index.tolist()[:n]

        return result, self.X[result + cols_rest] if return_all else result, self.X[result]

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
        Only the top features according to the feature importances computed by the algorithm will be selected.
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
