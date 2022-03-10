from sklearn.model_selection import train_test_split


def split_df(df, test_ratio=0.2, val_ratio=None, target=None, random_state=1337):
    """
    Split a dataset into training set and test set
    df -> (train, test)
       -> (X_train, X_test, y_train, y_test)
    :param df: a DataFrame to be split
    :param test_ratio: ratio of test set, 0-1
    :param val_ratio: ratio of validation set, 0-1
        split into (train, test) if not specified
        split into (train, val, test) if specified
    :param target:
        split into (train, test) if not specified
        split into (X_train, X_test, y_train, y_test) if specified
    :param random_state: random state
    """
    if target:
        if val_ratio:
            count = df.shape[0]
            val_count = int(count * val_ratio)
            test_count = int(count * test_ratio)
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            val = df[:val_count]
            test = df[val_count:(val_count + test_count)]
            train = df[(val_count + test_count):]
            val = val.sample(frac=1, random_state=random_state).reset_index(drop=True)
            test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
            train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
            X_train = train.drop(target, axis=1, inplace=False)
            X_val = val.drop(target, axis=1, inplace=False)
            X_test = test.drop(target, axis=1, inplace=False)
            y_train = train[target]
            y_val = val[target]
            y_test = test[target]
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X = df.drop(target, axis=1, inplace=False)
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_state)
            return X_train, X_test, y_train, y_test
    else:
        if val_ratio:
            count = df.shape[0]
            val_count = int(count * val_ratio)
            test_count = int(count * test_ratio)
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            val = df[:val_count]
            test = df[val_count:(val_count + test_count)]
            train = df[(val_count + test_count):]
            val = val.sample(frac=1, random_state=random_state).reset_index(drop=True)
            test = test.sample(frac=1, random_state=random_state).reset_index(drop=True)
            train = train.sample(frac=1, random_state=random_state).reset_index(drop=True)
            return train, val, test
        else:
            train, test = train_test_split(df, test_size=test_ratio, random_state=random_state)
            return train, test


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
