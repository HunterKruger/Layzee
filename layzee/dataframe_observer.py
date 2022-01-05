import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import shapiro, chi2_contingency, f_oneway

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

"""
EDA (exploratory data analysis) of a pd.DataFrame.
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
def describe_cat_col(df, col_name, top_n=None, plot_size=(8, 6), return_result=False, file_name=None):
    """
    Describe basic stats of a categorical column in a DataFrame.
    :param df: a DataFrame
    :param col_name: column name
    :param top_n: keep top N classes in count table, plot only top N classes
    :param plot_size: (x, y)
    :param return_result: return stats in a dict if True
    :param file_name: path + filename to save the plot
    """
    result = dict()
    result['Name'] = col_name
    result['Type'] = str(df[col_name].dtype)
    result['Rows'] = df.shape[0]
    result['Distinct'] = df[col_name].nunique()
    result['Mode'] = df[col_name].value_counts().index[0]
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

    if plot_size is not None:
        sns.set(rc={'figure.figsize': plot_size})
    plt.figure()
    # nan will not be plotted
    plot = sns.countplot(y=col_name, data=df, order=df[col_name].value_counts().nlargest(top_n).index)

    if file_name is not None:
        fig = plot.get_figure()
        fig.savefig(file_name)

    if return_result:
        return result, count_table_info


# test passed
def describe_num_col(df, col_name, plot_size=None, return_result=False, file_name=None):
    """
    Describe basic stats of a categorical column in a DataFrame.
    :param df: a DataFrame
    :param col_name: column name
    :param plot_size: (x, y)
    :param file_name: path + filename to save the plot
    :param return_result: return stats in a dict if True
    """
    q1 = df[col_name].quantile(q=0.25)
    q3 = df[col_name].quantile(q=0.75)
    iqr = q3 - q1
    upper_bound = 1.5 * iqr + q3
    lower_bound = q1 - 1.5 * iqr
    outliers = len(df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)])

    result = dict()
    result['Name'] = col_name
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
    shapiro_test = shapiro(df[col_name])
    print('Shapiro-Wilk statistic: ', shapiro_test.statistic)
    print('Significance level: 0.05')
    print('p-value: ', shapiro_test.pvalue)
    seg_lv = 0.05
    result['shapiro_stat'] = shapiro_test.statistic
    result['p_value'] = shapiro_test.pvalue
    if shapiro_test.pvalue >= seg_lv:
        print('<' + col_name + ' is normally distributed> cannot be rejected.')
        result['norm_dist'] = True
    else:
        print('<' + col_name + ' is normally distributed> can be rejected.')
        result['norm_dist'] = False
    print('-------------------------------------')

    if plot_size is not None:
        sns.set(rc={'figure.figsize': plot_size})

    plt.figure()
    fig, ax = plt.subplots(2, 1)
    sns.histplot(data=df, x=col_name, kde=True, ax=ax[0])
    sns.boxplot(x=df[col_name], ax=ax[1])

    if file_name is not None:
        fig.savefig(file_name)

    if return_result:
        return result


def describe_num_num(df, num_col1, num_col2, plot_size=None, return_result=False, file_name=None):
    """
    Observe two numerical columns.
    :param df: a DataFrame
    :param num_col1: 1st numerical column name
    :param num_col2: 2nd numerical column name
    :param plot_size: (x,y)
    :param file_name: to save plot
    :param return_result: return result in dict
    """
    result = {
        'col1': num_col1,
        'type1': df[num_col1].dtype,
        'col2': num_col2,
        'type2': df[num_col2].dtype
    }

    if plot_size is not None:
        sns.set(rc={'figure.figsize': plot_size})
    plt.figure()
    plot = sns.scatterplot(data=df, x=num_col1, y=num_col2)

    corr = df[[num_col1, num_col2]].corr()
    corr_value = corr[num_col1][1]
    print('Pearson correlation: ', corr_value)
    result['Pearson_corr'] = corr_value

    if file_name is not None:
        fig = plot.get_figure()
        fig.savefig(file_name)

    if return_result:
        return result


def describe_cat_cat(df, cat_col1, cat_col2, plot_size=None, return_result=False, file_name=None):
    """
    Observe two numerical columns.
    :param df: a DataFrame
    :param cat_col1: 1st categorical column
    :param cat_col2: 2nd categorical column
    :param return_result: return result in dict
    :param plot_size: (x,y)
    :param file_name: path + filename to save the plot
    """
    result = {'col1': cat_col1,
              'type1': df[cat_col1].dtype,
              'col2': cat_col2,
              'type2': df[cat_col2].dtype
              }

    cross_table = pd.crosstab(df[cat_col1], df[cat_col2], margins=False)

    if plot_size is not None:
        sns.set(rc={'figure.figsize': plot_size})
    plt.figure()
    plot = sns.heatmap(cross_table, annot=True, fmt="d", linewidths=.5, cmap=sns.cm.rocket_r, vmin=0)

    sig_lv = 0.05
    chi_value, p_value, deg_free, _ = chi2_contingency(cross_table)
    print('Chi-square statistic: ', chi_value)
    print('Degrees of freedom: ', deg_free)
    print('Significance level: ', sig_lv)
    print('p-value: ', p_value)
    result['Chi2'] = chi_value
    result['p_value'] = p_value
    result['deg_free'] = deg_free

    if p_value >= sig_lv:
        result['independent'] = True
        print('<' + cat_col1 + ' and ' + cat_col2 + ' are independent> cannot be rejected.')
    else:
        result['independent'] = False
        print('<' + cat_col1 + ' and ' + cat_col2 + ' are independent> can be rejected.')

    if file_name is not None:
        fig = plot.get_figure()
        fig.savefig(file_name)

    if return_result:
        return result


def describe_cat_num(df, num_col, cat_col,
                     plot_size=None, return_result=False, file_name=None):
    """
    :param df: a DataFrame
    :param num_col: a numerical column
    :param cat_col: a categorical column
    :param return_result: return result in dict
    :param plot_size: (x,y)
    :param file_name: path + filename to save the plot
    """
    result = {
        'col1': num_col,
        'type1': df[num_col].dtype,
        'col2': cat_col,
        'type2': df[cat_col].dtype
    }

    if plot_size is not None:
        sns.set(rc={'figure.figsize': plot_size})
    plt.figure()

    fig, ax = plt.subplots(2, 1)
    sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax[0])
    sns.violinplot(x=cat_col, y=num_col, data=df, ax=ax[1], inner="quartile")

    if file_name is not None:
        fig.savefig(file_name)

    uniques = df[cat_col].dropna().unique().tolist()
    list_unique = [df[num_col][df[cat_col] == x].tolist() for x in uniques]
    f_oneway_result = f_oneway(*list_unique)
    sig_lv = 0.05
    print('F oneway statistic: ', f_oneway_result.statistic)
    print('Significance level: ', sig_lv)
    print('p-value: ', f_oneway_result.pvalue)
    result['f_oneway_stat'] = f_oneway_result.statistic
    result['p_value'] = f_oneway_result.pvalue
    if f_oneway_result.pvalue > sig_lv:
        result['same_mean'] = True
        print('<' + num_col + ' has the same mean for each class> cannot be rejected.')
    else:
        result['same_mean'] = False
        print('<' + num_col + ' has the same mean for each class> can be rejected.')

    if return_result:
        return result


def auto_describe_col(df, col, plot_size=None, return_result=False, file_name=None):
    """
    Automatically describe a column in a DataFrame
    :param df: a DataFrame
    :param col: column name
    :param return_result: return result in dict
    :param plot_size: (x,y)
    :param file_name: path + filename to save the plot
    """
    if df[col].dtype == 'object':
        result, _ = describe_cat_col(df, col, plot_size=plot_size, return_result=return_result, file_name=file_name)
    else:
        result = describe_num_col(df, col, plot_size=plot_size, return_result=return_result, file_name=file_name)
    return result


def auto_describe_pair(df, col1, col2, plot_size=None, return_result=False, file_name=None):
    """
    Automatically describe a pair of columns in a DataFrame
    :param df: a DataFrame
    :param col1: 1st column
    :param col2: 2nd column
    :param return_result: return result in dict
    :param plot_size: (x,y)
    :param file_name: path + filename to save the plot
    """
    col_type1 = df[col1].dtype
    col_type2 = df[col2].dtype
    if col_type1 != 'object' and col_type2 != 'object':
        return describe_num_num(
            df=df, num_col1=col1, num_col2=col2,
            plot_size=plot_size, return_result=return_result, file_name=file_name
        )
    if col_type1 == 'object' and col_type2 == 'object':
        return describe_cat_cat(
            df=df, cat_col1=col1, cat_col2=col2,
            plot_size=plot_size, return_result=return_result, file_name=file_name
        )
    if col_type1 == 'object' and col_type2 != 'object':
        return describe_cat_num(
            df=df, cat_col=col1, num_col=col2,
            plot_size=plot_size, return_result=return_result, file_name=file_name
        )
    if col_type1 != 'object' and col_type2 == 'object':
        return describe_cat_num(
            df=df, cat_col=col2, num_col=col1,
            plot_size=plot_size, return_result=return_result, file_name=file_name
        )


# test passed
def missing_pattern(df, top_n=5, plot_size=None, file_name=None):
    """
    Check the missing patterns of a DataFrame
    :param df: a DataFrame
    :param top_n: the top n patterns to be displayed
    :param plot_size: (x,y)
    :param file_name: path + filename to save the plot
    """
    if plot_size is not None:
        sns.set(rc={'figure.figsize': plot_size})
    plt.figure()
    plot = sns.heatmap(df[df.columns[df.isnull().any()].tolist()].isnull(), yticklabels=False, cmap='hot_r', cbar=False)

    df_miss = df[df.columns[df.isnull().any()]].applymap(lambda x: '1' if pd.isna(x) else '0')
    row_miss = df_miss.apply(lambda x: '-'.join(x.values), axis=1)
    top_miss_rows_count = row_miss.value_counts().nlargest(top_n).tolist()
    top_miss_rows_pattern = row_miss.value_counts().index.tolist()[:top_n]
    headers = df.columns[df.isnull().any()].tolist()

    result_df = pd.DataFrame(data=[x.split('-') for x in top_miss_rows_pattern], columns=headers)
    result_df['PATTERN_COUNT'] = top_miss_rows_count
    result_df.set_index('PATTERN_COUNT', inplace=True)

    if file_name is not None:
        fig = plot.get_figure()
        fig.savefig(file_name)

    return result_df


# test passed
def correlation(df, method='pearson', k=None, plot_size=None, file_name=None):
    """
    Plot correlation between columns, pairs with correlations larger than the specified threshold will be returned.
    :param df: a DataFrame
    :param method: 'pearson', 'spearman', 'kendall'
    :param plot_size: (x,y)
    :param k:
        None, return corr matrix
        0~1, return pairs with abs(corr)>=threshold and corr matrix
        >1 in int, return pairs with top k corr and corr matrix
    :param file_name: path + filename to save the plot
    """
    corr_cols = df.columns.tolist()
    corr = df.corr(method)

    if plot_size is not None:
        sns.set(rc={'figure.figsize': plot_size})
    plt.figure()
    plot = sns.heatmap(corr, annot=True, linewidths=.5, cmap=sns.cm.vlag, vmin=-1, vmax=1)

    if file_name is not None:
        fig = plot.get_figure()
        fig.savefig(file_name)

    if k is None:
        return corr

    if 0 < k <= 1:
        result = []
        for i in range(len(corr_cols)):
            for j in range(i):
                corr_ij = corr.loc[corr_cols[i], corr_cols[j]]
                if abs(corr_ij) >= k:
                    result.append((corr_cols[i], corr_cols[j], corr_ij, abs(corr_ij)))
        return result, corr

    if k > 1:
        result = []
        for i in range(len(corr_cols)):
            for j in range(i):
                corr_ij = corr.loc[corr_cols[i], corr_cols[j]]
                result.append((corr_cols[i], corr_cols[j], corr_ij, abs(corr_ij)))
        sorted_result = sorted(result, key=lambda t: t[-1])[::-1]  # sort by abs(corr) in desc order
        return sorted_result[:int(k)], corr
