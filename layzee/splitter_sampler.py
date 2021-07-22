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
        return train.drop(target, axis=1).copy(), test.drop(target, axis=1).copy(), \
               train[target].copy(), test[target].copy()
