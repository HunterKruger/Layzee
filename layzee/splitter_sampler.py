import pandas as pd
from sklearn.model_selection import train_test_split


class SplitterSampler:

    def __init__(self):
        pass

    @staticmethod
    def split_df(df, test_ratio=0.2, target=None, random_state=1337):
        """
        Split a dataset into training set and test set
        df -> (train, test) or (X_train, X_test, y_train, y_test)
        :param df: a DataFrame to be split
        :param test_ratio: ratio of test set, 0-1.00 in decimal
        :param target:
            split into (train, test) if not specified
            split into (X_train, X_test, y_train, y_test) if specified
        :param random_state: random state
        """
        if target:
            X = df.drop(target, axis=1, inplace=False)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
            return X_train, X_test, y_train, y_test
        else:
            train, test = train_test_split(df, test_size=test_ratio, random_state=random_state)
            return train, test

    @staticmethod
    def split_Xy(X, y, test_ratio=0.2, random_state=1337):
        """
        (X,y) -> (X_train, X_test, y_train, y_test)
        :param X: features, 2d
        :param y: target, 1d
        :param test_ratio: ratio of test set, 0-1
        :param random_state: random state
        """
        return train_test_split(X, y, test_size=test_ratio, random_state=random_state)

    @staticmethod
    def split_train_test(train, test, target):
        """
        (train,test) -> (X_train, X_test, y_train, y_test)
        :param target: target column
        :param train: training set with features and target
        :param test: test set with features and target
        """
        return train.drop(target, axis=1).copy(), \
               test.drop(target, axis=1).copy(), \
               train[target].copy(), \
               test[target].copy()

    @staticmethod
    def sampler(df, records, groupby=None, random_state=1337):
        """
        sample a DataFrame with a given count
        :param df: a DataFrame
        :param records: number of rows to be sampled if >= 1
                        proportion of rows to be sampled if > 0 and < 1
        :param groupby: sampling proportionally based on a categorical column's distribution
        :param random_state: random state
        :return:
        """
        if groupby is None:
            if records >= 1:
                return df.sample(n=records, random_state=random_state)
            if 0 < records < 1:
                return df.sample(frac=records, random_state=random_state)
        else:
            grouped = df.groupby(groupby)
            if records >= 1:
                grouped.apply(lambda x: x.sample(n=records, random_state=random_state))
            if 0 < records < 1:
                grouped.apply(lambda x: x.sample(frac=records, random_state=random_state))
            return grouped.reset_index(level=0, drop=True).sort_index(ascending=True)
