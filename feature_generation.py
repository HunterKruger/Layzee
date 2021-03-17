import pandas as pd
from pandas import DataFrame
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures


class FeatureGeneration:

    def __init__(self):
        pass

    # test passed
    @staticmethod
    def pairwise_linear_combinations(df: DataFrame, num_cols='auto', return_cols=False, drop_origin=False):
        """
        Generate A+B and A-B features
        :param df: DataFrame
        :param num_cols: list of numerical columns to generate new features
        :param return_cols: choose 'True' to return generated feature list
        :param drop_origin: choose 'True' to drop 'num_cols' in 'df'
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

        return df, new_cols if return_cols else df

    # test passed
    @staticmethod
    def pairwise_polynomial_combinations(df: DataFrame, num_cols='auto', return_cols=False, drop_origin=False):
        """
        Generate A*B features
        :param df: DataFrame
        :param num_cols: list of numerical columns to generate new features
        :param return_cols: choose 'True' to return generated feature list
        :param drop_origin: choose 'True' to drop 'num_cols' in 'df'
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
        return df, new_cols if return_cols else df

    @staticmethod
    def explicit_pairwise_interactions(df: DataFrame, col1, col2, return_cols=False, drop_origin=False):
        """
        This generator generates pairwise interactions between 2 features:
        Numerical features will be multiplied
        Numerical and categorical features will produce a dummies multiplied by the numerical feature.
        Two categorical features will produce dummies in the cross-product of the two features
        :param df: DataFrame
        :param col1: specify the 1st column
        :param col2: specify the 2nd column
        :param return_cols: choose 'True' to return generated feature list
        :param drop_origin: choose 'True' to drop 'num_cols' in 'df'
        :return:
        """
        if df[col1].dtype == 'O' and df[col2].dtype == 'O':
            pass
        if df[col1].dtype != 'O' and df[col2].dtype == 'O':
            pass
        if df[col1].dtype == 'O' and df[col2].dtype != 'O':
            pass
        if df[col1].dtype != 'O' and df[col2].dtype != 'O':
            pass
