from pandas import DataFrame
from sklearn.cluster import KMeans


# test passed
def pairwise_linear_combinations(df: DataFrame, num_cols='auto', drop_origin=False):
    """
    Generate A+B and A-B features for numerical features
    :param df: DataFrame
    :param num_cols:
        list of numerical columns to generate new features;
        automatically choose all numerical features if 'auto'
    :param drop_origin: choose 'True' to drop 'num_cols' in 'df'
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
def pairwise_polynomial_combinations(df: DataFrame, num_cols='auto', drop_origin=False):
    """
    Generate A*B features for numerical features
    :param df: DataFrame
    :param num_cols:
        list of numerical columns to generate new features;
        automatically choose all numerical features if 'auto'
    :param drop_origin: choose 'True' to drop 'num_cols' in 'df'
    :return
        df: 'df' with generated new features
        new_cols: list of generated new features
    """
    if num_cols == 'auto':
        num_cols = df.select_dtypes(include='number').columns.tolist()

    pairs = [(a, b) for idx, a in enumerate(num_cols) for b in num_cols[idx + 1:]]
    for i, j in pairs:
        df[i + '*' + j] = df[i] + df[j]
    if drop_origin:
        df.drop(num_cols, axis=1, inplace=True)
    return df


def kmeans_featurization(df, cols, n_clusters=3, df2=None):
    """
    Feed features and target into KMeans to generate a new feature that indicates the cluster where it belongs to.
    :param df: DataFrame, training set, including features and the target
    :param df2: DataFrame, test set, including features and the target
    :param n_clusters: number of clusters in kmeansz
    :param cols: list of encoded columns to be used;
    """
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(df[cols])
    df['cluster'] = kmeans.predict(df)
    if df2 is None:
        return df
    else:
        df2['cluster'] = kmeans.predict(df2)
        return df, df2
