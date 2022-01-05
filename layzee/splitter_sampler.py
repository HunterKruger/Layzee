from sklearn.model_selection import train_test_split


def split_df(df, test_ratio=0.2, target=None, random_state=1337):
    """
    Split a dataset into training set and test set
    df -> (train, test)
       -> (X_train, X_test, y_train, y_test)
    :param df: a DataFrame to be split
    :param test_ratio: ratio of test set, 0-1
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


def split_Xy(X, y, test_ratio=0.2, random_state=1337):
    """
    (X,y) -> (X_train, X_test, y_train, y_test)
    :param X: features, 2d
    :param y: target, 1d
    :param test_ratio: ratio of test set, 0-1
    :param random_state: random state
    """
    return train_test_split(X, y, test_size=test_ratio, random_state=random_state)


def split_train_test(train, test, target):
    """
    (train,test) -> (X_train, X_test, y_train, y_test)
    :param target: target column
    :param train: training set with features and target
    :param test: test set with features and target
    """
    return train.drop(target, axis=1), test.drop(target, axis=1), train[target], test[target]


def sampler(df, n, col=None, random_state=1337):
    """
    Random/Stratified sampling
    :param df: a DataFrame
    :param n: samples or fraction
    :param col: column name for stratified sampling
    :param random_state: random state
    """
    if 0 < n < 1 and col is None:
        return df.sample(n=n, random_state=random_state)
    if n > 1 and col is None:
        return df.sample(frac=n, random_state=random_state)
    if 0 < n < 1 and col is not None:
        new_df, _ = train_test_split(df, test_size=1 - n, stratify=df[[col]], random_state=random_state)
        return new_df
    if n > 1 and col is not None:
        new_df, _ = train_test_split(df, test_size=(len(df) - n) / len(df), stratify=df[[col]],
                                     random_state=random_state)
        return new_df
