import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

"""
For EDA (exploratory data analysis) of a pd.DataFrame.
All test passed.
"""


# test passed
def read_df_info(df, return_result=False):
    """
    Print general stats of a DataFrame.
    :param df: a DataFrame
    :param return_result: return stats in a dict if True
    """
    print('------------------')
    print(str(df.shape[0]) + ' rows, ' + str(df.shape[1]) + ' columns.')
    print('------------------')
    temp = pd.concat([df.dtypes, df.count()], axis=1)
    temp.columns = ['dtype', 'count']
    uni = pd.DataFrame()
    uni['unique'] = df.nunique()
    uni['unique%'] = uni['unique'] / df.shape[0]
    ms = pd.DataFrame()
    ms['missing'] = df.isnull().sum()
    ms['missing%'] = ms['missing'] / df.shape[0]
    temp2 = uni.join(other=ms, how='left')
    result = temp.join(other=temp2, how='left')
    result.drop('count', axis=1, inplace=True)
    print(result)
    print('------------------')
    if return_result:
        return result


# test passed
def describe_cat_col(df, col_name, top_n=None, plot_size=(8, 6), return_result=False):
    """
    Describe basic stats of a categorical column in a DataFrame.
    :param df: a DataFrame
    :param col_name: column name
    :param top_n: keep top N classes in count table, plot only top N classes
    :param plot_size: size of plot, tuple
    :param return_result: return stats in a dict if True
    """

    result = dict()
    result['Type'] = str(df[col_name].dtype)
    result['Rows'] = df.shape[0]
    result['Distinct'] = df[col_name].nunique()
    result['Missing'] = df[col_name].isnull().sum()
    result['Missing%'] = df[col_name].isnull().sum() / df.shape[0]

    print('-------------------------------------')
    for k, v in result.items():
        print(str(k) + ': ' + str(v))
    print('-------------------------------------')

    if top_n is None:
        top_n = len(df[col_name].value_counts())

    print('Top ' + str(top_n) + ' values:')  # nan will not be counted
    count_table_info = pd.concat([
        df[col_name].value_counts().nlargest(top_n),
        df[col_name].value_counts(normalize=True).nlargest(top_n),
        df[col_name].value_counts(normalize=True).nlargest(top_n).cumsum()],
        axis=1
    )

    count_table_info.columns = ['Count', '%', 'Cum.%']
    count_table_info = count_table_info.sort_values(by='Count', ascending=False)
    count_table_info.reset_index(inplace=True)
    count_table_info.columns = [col_name, 'Count', '%', 'Cum.%']
    print(count_table_info)
    print('-------------------------------------')
    # nan will not be plotted
    plt.subplots(figsize=plot_size)
    sns.countplot(y=col_name, data=df, order=df[col_name].value_counts().nlargest(top_n).index)

    if return_result:
        return result, count_table_info


# test passed
def describe_num_col(df, col_name, return_result=False):
    """
    Describe basic stats of a categorical column in a DataFrame.
    :param df: a DataFrame
    :param col_name: column name
    :param return_result: return stats in a dict if True
    """
    q1 = df[col_name].quantile(q=0.25)
    q3 = df[col_name].quantile(q=0.75)
    iqr = q3 - q1
    upper_bound = 1.5 * iqr + q3
    lower_bound = q1 - 1.5 * iqr
    outliers = len(df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)])

    result = dict()
    result['Type'] = df[col_name].dtype
    result['Rows'] = df.shape[0]
    result['Min'] = df[col_name].min()
    result['Max'] = df[col_name].max()
    result['Mean'] = df[col_name].mean()
    result['Median'] = df[col_name].median()
    result['Mode'] = df[col_name].value_counts().index[0]
    result['StdDev'] = df[col_name].std()
    result['Distinct'] = df[col_name].nunique()
    result['Sum'] = df[col_name].sum()
    result['Missing'] = df[col_name].isnull().sum()
    result['Missing%'] = df[col_name].isnull().sum() / df.shape[0]
    result['Skewness'] = df[col_name].skew()
    result['Kurtosis'] = df[col_name].kurtosis()
    result['Outliers'] = outliers
    result['Outliers%'] = outliers / df.shape[0]
    result['Q1'] = q1
    result['Q3'] = q3
    result['IQR'] = iqr
    result['Down'] = lower_bound
    result['Up'] = upper_bound

    print('-------------------------------------')
    for k, v in result.items():
        print(str(k) + ': ' + str(v))
    print('-------------------------------------')
    fig, ax = plt.subplots(2, 1)
    sns.histplot(data=df, x=col_name, kde=True, ax=ax[0])
    sns.boxplot(x=df[col_name], ax=ax[1])

    if return_result:
        return result


# test passed
def missing_pattern(df, top_n=5):
    """
    Check the missing patterns of a DataFrame
    :param df: a DataFrame
    :param top_n: the top n patterns to be displayed
    """
    sns.heatmap(df[df.columns[df.isnull().any()].tolist()].isnull(), yticklabels=False, cmap='hot_r', cbar=False)
    df_miss = df[df.columns[df.isnull().any()]].applymap(lambda x: '1' if pd.isna(x) else '0')
    row_miss = df_miss.apply(lambda x: '-'.join(x.values), axis=1)
    print(df.columns[df.isnull().any()].tolist())
    print(row_miss.value_counts().nlargest(top_n))


# test passed
def correlation(df, method='pearson', threshold=1, plot_size_x=7, plot_size_y=6):
    """
    Plot correlation between columns, pairs with correlations larger than the specified threshold will be returned.
    :param df: a DataFrame
    :param method: 'pearson', 'spearman', 'kendall'
    :param plot_size_x: plot size in x axis
    :param plot_size_y: plot size in y axis
    :param threshold: 0~1, return pairs with abs(corr)>=threshold if specified
    """
    plt.subplots(figsize=(plot_size_x, plot_size_y))
    corr_cols = df.columns.tolist()
    corr = df.corr(method)
    sns.heatmap(corr, annot=True, linewidths=.5, cmap=sns.cm.vlag, vmin=-1, vmax=1)
    plt.show()

    result = []
    for i in range(len(corr_cols)):
        for j in range(i):
            corr_ij = corr.loc[corr_cols[i], corr_cols[j]]
            if abs(corr_ij) >= threshold:
                result.append((corr_cols[i], corr_cols[j], corr_ij))
    return result
