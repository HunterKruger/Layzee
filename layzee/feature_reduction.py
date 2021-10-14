from abc import abstractmethod
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.decomposition import PCA


def mode_based(X, threshold, cat_cols='auto'):
    """
    Filter categorical features based on a specific threshold of mode proportion
    :param X: df without target, encoded
    :param threshold: threshold of mode proportion, 0-1.00 in decimal
    :param cat_cols: list of str, categorical features
                     choose 'auto' for auto detection
    :return: X reduced
    """
    if cat_cols == 'auto':
        cat_cols = X.select_dtypes('object').columns.tolist()
    to_drop = []
    for col in cat_cols:
        if X[col].value_counts(normalize=True).nlargest().values[0] > threshold:
            to_drop.append(col)
    return X.drop(to_drop, axis=1)


# test passed
def pca(X, n):
    """
    The feature space dimension will be reduced using Principal Component Analysis.
    Only the top principal components will be selected.
    This method will generate non-interpretable feature names as its output.
    The model may be performant, but will not be interpretable.
    :param X: df without target, encoded
    :param n:
        if int, number of components
        if  0 < decimal < 1, select the number of components such that the amount of variance that needs to be
            explained is greater than the percentage specified by n_components
    :return
        reduced X
    """
    result = PCA(n_components=n).fit_transform(X)
    nb_cols = result.shape[1]
    cols = ['pc' + str(i) for i in range(nb_cols)]
    return pd.DataFrame(result, columns=cols)


# test passed
def tree_based_reg(X, y, n_keep, n_trees, depth, n_jobs=-1):
    """
    Only for regression case!
    This creates a Random Forest model to predict the target.
    Only the top features according to the feature importance computed by the algorithm will be selected.
    :param X: df without target, encoded
    :param y: target col
    :param n_keep: number of features to keep
    :param n_trees: number of trees in Random Forest
    :param depth: tree depth in Random Forest
    :param n_jobs: multi-core
    :return X with only top features
    """
    model = RandomForestRegressor(n_estimators=n_trees, max_depth=depth, n_jobs=n_jobs)
    model.fit(X, y)
    feature_importance = sorted(zip(X.columns.to_list(), model.feature_importances_), reverse=True)
    top_list = [feature for feature, score in feature_importance]
    if n_keep > len(top_list):
        n_keep = len(top_list)
    return X[top_list[:n_keep]]


# test passed
def lasso_reg(X, y, l1='auto'):
    """
    Only for regression case!
    This creates a LASSO model to predict the target, using 3-fold cross-validation to select the best value
    of the regularization term. Only the features with nonzero coefficients will be selected.
    :param X: df without target, encoded
    :param y: target col
    :param l1: list of floats, for l1 penalty
    :return: X with only selected features
    """
    if l1 == 'auto':
        l1 = [0.10, 0.1, 1, 10, 100]
    model = LassoCV(cv=3, alphas=l1).fit(X, y)
    result_series = pd.Series(list(model.coef_), index=X.columns)
    result_features = result_series[result_series != 0].index.tolist()
    return X[result_features]


def pearson_corr(X, y, n_keep, num_cols):
    """
    :param X: df without target, encoded
    :param y: target col
    :param n_keep: number of numerical columns to be kept
    :param num_cols: list of numerical features in X to be calculated with y
    :return: X with selected num_cols and non-num_cols together
    """
    if n_keep >= len(num_cols):
        raise ValueError("n must smaller than number of numerical cols!")
    else:
        X_temp = X[num_cols]
    rest_cols = list(set(X.columns.tolist()) - set(num_cols))
    all_data = pd.concat([X_temp, y], axis=1)
    selected_cols = all_data.corr('pearson').iloc[:-1, -1].sort_values(ascending=False).index.tolist()[:n_keep]
    return X[selected_cols + rest_cols]


# test passed
def tree_based_cls(X, y, n_keep, n_trees, depth, n_jobs=-1):
    """
    Only for classification case!
    This creates a Random Forest model to predict the target.
    Only the top features according to the feature importance computed by the algorithm will be selected.
    :param X: df without target, encoded
    :param y: target col
    :param n_keep: number of features to keep
    :param n_trees: number of trees in Random Forest
    :param depth: tree depth in Random Forest
    :param n_jobs: multi-core
    :return DataFrame with only top features
    """
    model = RandomForestClassifier(n_estimators=n_trees, max_depth=depth, n_jobs=n_jobs)
    model.fit(X, y)
    ft_imp = sorted(zip(X.columns.to_list(), model.feature_importances_), reverse=True)
    top_list = [feature for feature, score in ft_imp]
    if n_keep > len(top_list):
        n_keep = len(top_list)
    return X[top_list[:n_keep]]


# test passed
def lasso_cls(X, y, l1='auto', n_jobs=-1):
    """
    Only for classification case!
    This creates a LASSO model to predict the target, using 3-fold cross-validation to select the best value
    of the regularization term. Only the features with nonzero coefficients will be selected.
    :param l1: list of floats, for l1 penalty
    :param X: df without target, encoded
    :param y: target col
    :param n_jobs: multi-core
    :return: DataFrame with only selected features
    """
    if l1 is 'auto':
        l1 = [0.10, 0.1, 1, 10, 100]
    model = LogisticRegressionCV(
        penalty='l1', cv=3, solver='liblinear', Cs=[1 / x for x in l1],
        class_weight='balanced', n_jobs=n_jobs).fit(X, y)
    result_series = pd.Series(list(model.coef_[0]), index=X.columns)
    result_features = result_series[result_series != 0].index.tolist()
    return X[result_features]
