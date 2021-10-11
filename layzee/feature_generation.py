import pandas as pd
from pandas import DataFrame
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures


# test passed
def pairwise_linear_combinations(df: DataFrame, num_cols='auto', drop_origin=False):
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
    # find all pairs
    pairs = [(a, b) for idx, a in enumerate(num_cols) for b in num_cols[idx + 1:]]
    for i, j in pairs:
        df[i + '+' + j] = df[i] + df[j]
        df[i + '-' + j] = df[i] - df[j]
    if drop_origin:
        df.drop(num_cols, axis=1, inplace=True)
    return df


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
