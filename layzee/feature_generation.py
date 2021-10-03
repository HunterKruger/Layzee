import pandas as pd
from pandas import DataFrame
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures


# test passed
def pairwise_linear_combinations(df: DataFrame, num_cols='auto', return_cols=False, drop_origin=False):
    """
    Generate A+B and A-B features for numerical features
    :param df: DataFrame
    :param num_cols:
        list of numerical columns to generate new features;
        automatically choose all numerical features if 'auto'
    :param drop_origin: choose 'True' to drop 'num_cols' in 'df'
    :param return_cols: choose 'True' to return generated feature in a list
    :return
        df: 'df' with generated new features
        new_cols: list of generated new features
    """
    if num_cols == 'auto':
        num_cols = df.select_dtypes(include='number').columns.tolist()

    new_cols = []
    # find all pairs
    pairs = [(a, b) for idx, a in enumerate(num_cols) for b in num_cols[idx + 1:]]
    print(pairs)
    for i, j in pairs:
        df[i + '+' + j] = df[i] + df[j]
        df[i + '-' + j] = df[i] - df[j]
        new_cols.append(i + '+' + j)
        new_cols.append(i + '-' + j)
    if drop_origin:
        df.drop(num_cols, axis=1, inplace=True)
    return df, new_cols if return_cols is True else df


# test passed
def pairwise_polynomial_combinations(df: DataFrame, num_cols='auto', return_cols=False, drop_origin=False):
    """
    Generate A*B features for numerical features
    :param df: DataFrame
    :param num_cols:
        list of numerical columns to generate new features;
        automatically choose all numerical features if 'auto'
    :param drop_origin: choose 'True' to drop 'num_cols' in 'df'
    :param return_cols: choose 'True' to return generated feature in a list
    :return
        df: 'df' with generated new features
        new_cols: list of generated new features
    """
    if num_cols == 'auto':
        num_cols = df.select_dtypes(include='number').columns.tolist()
    new_cols = []
    # find all pairs
    pairs = [(a, b) for idx, a in enumerate(num_cols) for b in num_cols[idx + 1:]]
    print(pairs)
    for i, j in pairs:
        df[i + '*' + j] = df[i] + df[j]
        new_cols.append(i + '*' + j)
    if drop_origin:
        df.drop(num_cols, axis=1, inplace=True)
    return df, new_cols if return_cols is True else df


def explicit_pairwise_interactions(df: DataFrame, col1, col2, return_cols=False, drop_origin=False):
    """
    This generator generates pairwise interactions between 2 features:
    Numerical features will be multiplied
    Numerical and categorical features will produce a dummies multiplied by the numerical feature.
    Two categorical features will produce dummies in the cross-product of the two features
    :param df: DataFrame
    :param col1: specify the 1st column
    :param col2: specify the 2nd column
    :param return_cols: choose 'True' to return generated feature in a list
    :param drop_origin: choose 'True' to drop 'num_cols' in 'df'
    :return:
    """
    new_cols = []
    if df[col1].dtype == 'O' and df[col2].dtype == 'O':
        dummies1 = pd.get_dummies(df[col1], prefix=col1, dummy_na=True)
        dummies2 = pd.get_dummies(df[col2], prefix=col2, dummy_na=True)
        pass
    if df[col1].dtype != 'O' and df[col2].dtype == 'O':
        pass
    if df[col1].dtype == 'O' and df[col2].dtype != 'O':
        pass
    if df[col1].dtype != 'O' and df[col2].dtype != 'O':
        df[col1 + '*' + col2] = df[col1] * df[col2]
        new_cols.append(col1 + '*' + col2)

    if drop_origin:
        df.drop([col1, col2], axis=1, inplace=True)
    return df, new_cols if return_cols else df


def explicit_pairwise_interactions_batch(df: DataFrame, col_pairs, return_cols=False, drop_origin=False):
    """
    This generator generates pairwise interactions between several pairs of 2 features.
    Numerical features will be multiplied.
    Numerical and categorical features will produce a dummies multiplied by the numerical feature.
    Two categorical features will produce dummies in the cross-product of the two features
    :param df: DataFrame
    :param col_pairs: list of tuples, columns in a tuple will be used to generate new features
        eg: [(col1,col2),(col3,col4)]
    :param return_cols: choose 'True' to return generated feature list
    :param drop_origin: choose 'True' to drop 'num_cols' in 'df'
    :return:
    """
    for col1, col2 in col_pairs:
        explicit_pairwise_interactions(df, col1, col2, return_cols, drop_origin)


@staticmethod
def kmeans_featurization(df1, df2=None, cols='all', one_hot=None):
    """
    Feed features and target into KMeans to generate a new feature that indicates the cluster where it belongs to.
    :param df1: DataFrame, training set, including features and the target
    :param df2: DataFrame, test set, including features and the target
    :param one_hot: one-hot encode the new kmeans feature if True
    :param cols: list of encoded columns to be used; 'all' for all columns;
    :return:
        df1 with the new feature
        df2 with the new feature
    """
    if cols != 'all':
        df1 = df1[cols]
        df2 = df2[cols]

    pass
