import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder


def imputer(df, col, method='mode', groupby=None):
    """
    Impute missing values in a column
    :param df: a dataframe
    :param groupby: -> str or list of str:
            group by one or several categorical columns to calculate the values for imputing
    :param col: column name
    :param method: choose the method {'mean', 'mode', 'median'} for imputing, a customized constant is allowed
    """

    if groupby is None:
        if method == 'mean':
            temp = df[col].mean()
        elif method == 'median':
            temp = df[col].median()
        elif method == 'mode':
            temp = df[col].mode()[0]
        else:
            temp = method

        df[col].fillna(temp, inplace=True)

    else:
        if method == 'mean':
            df[col] = df.groupby(groupby)[col].apply(lambda x: x.fillna(x.mean()))
        elif method == 'median':
            df[col] = df.groupby(groupby)[col].apply(lambda x: x.fillna(x.median()))
        elif method == 'mode':
            df[col] = df.groupby(groupby)[col].apply(lambda x: x.fillna(x.mode()[0]))
        else:
            print("Please set <groupby> to None!")

    return df


def auto_imputers(df, df2, cat_cols=None, num_cols=None, cat_method='mode', num_method='median'):
    """
    Impute missing values in several columns at a time, automatically detect categorical and numerical features
    :param df: a dataframe
    :param df2: a dataframe
    :param cat_cols: list of categorical columns, automatically detect if None
    :param num_cols: list of numerical columns, automatically detect if None
    :param cat_method: imputing method for categorical features
    :param num_method: imputing method for numerical features
    """
    cat_cols = df.select_dtypes('object').columns.tolist() if cat_cols is None else cat_cols
    num_cols = df.select_dtypes('number').columns.tolist() if num_cols is None else num_cols
    for col in cat_cols:
        df = imputer(df, col, cat_method)
    for col in num_cols:
        df = imputer(df, col, num_method)

    return df


def keep_top_n(df, col, N, include_nan=False, replacer=np.nan):
    """
    Merge long tail to a specified value
    :param df: a dataframe
    :param col: column name, should be categorical
    :param N: keep top N class if N is integer(N>=1);
              keep classes whose percentage is higher than N if N is decimal(0<N<1)
    :param include_nan: include nan when counting top N classes
    :param replacer: the value to replace long tail
    """
    top_n = []
    if include_nan is True and N >= 1:
        top_n = df[col].value_counts(dropna=False).index.tolist()[:N]
    if include_nan is False and N >= 1:
        top_n = df[col].value_counts().index.tolist()[:N]
    if include_nan is True and N < 1:
        rank = df[col].value_counts(dropna=False, normalize=True)
        real_n = sum(i > N for i in rank.values.tolist())
        top_n = df[col].value_counts(dropna=False).index.tolist()[:real_n]
    if include_nan is False and N < 1:
        rank = df[col].value_counts(normalize=True)
        real_n = sum(i > N for i in rank.values.tolist())
        top_n = df[col].value_counts().index.tolist()[:real_n]
    df[col] = df[col].apply(lambda x: x if x in top_n else replacer)
    return df


def handle_outlier(df, col, drop=False):
    """
    Handle outliers in a numerical column;
    :param df: a dataframe
    :param col: column name, must be numerical
    :param drop:
        False: outliers replaced by missing value
        True: drop rows with outliers
    """
    q1 = df[col].quantile(q=0.25)
    q3 = df[col].quantile(q=0.75)
    iqr = q3 - q1
    upper_bound = 1.5 * iqr + q3
    lower_bound = q1 - 1.5 * iqr

    if drop:
        df = df[(df[col] <= upper_bound) & (df[col] >= lower_bound)]
    else:
        df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan

    return df


def outlier_encoder(df, col, drop_origin=True):
    """
    Encode outliers in this column by col_upper, col_lower, col_inter;
    Should not contain missing values, or they will be imputed by median;
    :param df: a dataframe
    :param col: column name, must be numerical
    :param drop_origin: drop origin column
    """

    df[col].fillna(df[col].median(), inplace=True)
    q1 = df[col].quantile(q=0.25)
    q3 = df[col].quantile(q=0.75)
    iqr = q3 - q1
    upper_bound = 1.5 * iqr + q3
    lower_bound = q1 - 1.5 * iqr

    df[col + '_lower'] = df[col].apply(lambda x: 1 if x < lower_bound else 0)
    df[col + '_upper'] = df[col].apply(lambda x: 1 if x > upper_bound else 0)
    df[col + '_inner'] = df[col].apply(lambda x: df[col].median() if x > upper_bound or x < lower_bound else x)

    if drop_origin:
        df.drop(col, axis=1, inplace=True)

    return df


def get_skewness(df, col):
    """
    Exclude outer 3% data
    :param df: a dataframe
    :param col: column name, must be numerical
    :return: skewness
    """
    q_015, q_985 = df[col].quantile([0.015, 0.985])
    return df[col][(col >= q_015) & (col <= q_985)].skew()


def handle_skewness(df, col, threshold=1, drop_origin=True):
    """
    log(1+x) transformation if col>0 and -1<skewness<1
    :param df: a dataframe
    :param col: column name, must be numerical
    :param threshold: for skewness
    :param drop_origin: drop origin column
    """
    skewness = get_skewness(df, col)
    print('Skewness: ' + str(skewness))

    if df[col].min() < 0:
        print('Make sure the min value is positive!')
        return

    if -1 * threshold <= skewness <= threshold:
        print('No need of transformation')
    else:
        min_value = df[col].min()
        if min_value < 0:
            df[col + '_log1x'] = df[col] + min_value  # move within x-axis
        df[col + '_log1x'] = np.log1p(df[col])
        print('Skewness: ' + get_skewness(df, col + '_log1x'))
        if drop_origin:
            df.drop(col, axis=1, inplace=True)

    return df


def general_encoder(df, num_cols='auto', ordinal_cols=None, one_hot_cols='auto', drop=None, return_encoders=False):
    """
    A general encoder to transform numerical, categorical and ordinal features；
    :param df: a dataframe
    :param drop: for one-hot encoded features
            - None : retain all features (the default)
            - 'first' : drop the first category in each feature. If only one
                        category is present, the feature will be dropped entirely
            - 'if_binary' : drop the first category in each feature with two
                            categories. Features with 1 or more than 2 categories are
                            left intact
    :param num_cols: list of numerical columns to be encoded;
                     select all numerical columns if 'auto'
    :param one_hot_cols: list of categorical columns to be one-hot encoded;
                         select all categorical columns if 'auto'
    :param ordinal_cols: list of ordinal columns to be ordinal encoded
    :param return_encoders: return 3 types of encoders
    """

    if drop is None:
        ohe = OneHotEncoder(handle_unknown='ignore')
    else:
        ohe = OneHotEncoder(handle_unknown='error', drop=drop)
    ss = StandardScaler()
    oe = OrdinalEncoder()

    if one_hot_cols is None:
        one_hot_cols = []
        df1_cat = pd.DataFrame()
    else:
        if one_hot_cols == 'auto':
            one_hot_cols = df.select_dtypes(include='object').columns.tolist()
        cls_cat1 = ohe.fit_transform(df[one_hot_cols]).toarray()
        df1_cat = pd.DataFrame(data=cls_cat1, columns=ohe.get_feature_names(one_hot_cols).tolist())

    if num_cols is None:
        num_cols = []
        df1_num = pd.DataFrame()
    else:
        if num_cols == 'auto':
            num_cols = df.select_dtypes(include='number').columns.tolist()
        cls_num1 = ss.fit_transform(df[num_cols])
        df1_num = pd.DataFrame(cls_num1, columns=num_cols)

    if ordinal_cols is None:
        ordinal_cols = []
        df1_ord = pd.DataFrame()
    else:
        cls_ord1 = oe.fit_transform(df[ordinal_cols])
        df1_ord = pd.DataFrame(cls_ord1, columns=ordinal_cols)

    cls_rest = list(set(df.columns.tolist()) - set(num_cols) - set(one_hot_cols) - set(ordinal_cols))

    df_encoded = pd.concat(
        [df1_num.reset_index(drop=True), df1_cat.reset_index(drop=True), df1_ord.reset_index(drop=True),
         df[cls_rest].copy().reset_index(drop=True)], axis=1)

    if return_encoders:
        return df_encoded, ohe, ss, oe
    else:
        return df_encoded
