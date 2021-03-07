import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class DataframeObserver:

    def __init__(self):
        pass

    # test passed
    @staticmethod
    def read_df_info(df, return_result=False):
        print('------------------')
        print(str(df.shape[0]) + ' rows, ' + str(df.shape[1]) + ' columns.')
        print('------------------')
        temp = pd.concat([df.dtypes, df.count()], axis=1)
        temp.columns = ['Dtype', 'Count']
        uni = pd.DataFrame()
        uni['unique'] = df.nunique()
        uni['unique%'] = uni['unique'] / df.shape[0]
        ms = pd.DataFrame()
        ms['missing'] = df.isnull().sum()
        ms['missing%'] = ms['missing'] / df.shape[0]
        temp2 = uni.join(other=ms, how='left')
        result = temp.join(other=temp2, how='left')
        print(result)
        print('------------------')
        if return_result:
            return result

    # test passed
    @staticmethod
    def describe_cat_col(df, col_name, top_n=None, plot_size_x=8, plot_size_y=6, return_result=False):
        """
        Describe basic stats of a categorical column in a DataFrame
        df: a DataFrame
        col_name: column name
        top_N: keep top N classes in count table, plot only top N classes
        plot_size_x: size of plot in x-axis
        plot_size_y: size of plot in y-axis
        """
        print('-------------------------------------')
        print('Type: ' + str(df[col_name].dtype))
        print('Rows: ' + str(df.shape[0]))
        print('Distinct: ' + str(df[col_name].nunique()))
        print('Missing: ' + str(df[col_name].isnull().sum()))
        print('Missing%: ' + str(df[col_name].isnull().sum() / df.shape[0]))
        print('-------------------------------------')

        if top_n is None:
            top_n = len(df[col_name].value_counts())

        print('Top ' + str(top_n) + ' values:')  # nan will not be counted
        count_table_info = pd.concat([
            df[col_name].value_counts().nlargest(top_n),
            df[col_name].value_counts(normalize=True).nlargest(top_n),
            df[col_name].value_counts(normalize=True).nlargest(top_n).cumsum()],
            axis=1)

        count_table_info.columns = ['Count', '%', 'Cum.%']
        count_table_info = count_table_info.sort_values(by='Count', ascending=False)
        count_table_info.reset_index(inplace=True)
        count_table_info.columns = [col_name, 'Count', '%', 'Cum.%']
        print(count_table_info)
        print('-------------------------------------')
        # nan will not be plotted
        fig, ax = plt.subplots(figsize=(plot_size_x, plot_size_y))
        sns.countplot(y=col_name, data=df, order=df[col_name].value_counts().nlargest(top_n).index)

        if return_result:
            result = dict()
            result['Type'] = str(df[col_name].dtype)
            result['Rows'] = df.shape[0]
            result['Distinct'] = df[col_name].nunique()
            result['Missing'] = df[col_name].isnull().sum()
            result['Missing%'] = df[col_name].isnull().sum() / df.shape[0]
            return result

    # test passed
    @staticmethod
    def describe_num_col(df, col_name, return_result=False):
        """
        Describe basic stats of a categorical column in a DataFrame
        col_name: column name
        """

        print('-------------------------------------')
        print('Type: ' + str(df[col_name].dtype))
        print('Rows: ' + str(df.shape[0]))
        print('Min: ' + str(df[col_name].min()))
        print('Max: ' + str(df[col_name].max()))
        print('Mean: ' + str(df[col_name].mean()))
        print('Median: ' + str(df[col_name].median()))
        print('Mode: ' + str(df[col_name].value_counts().index[0]))
        print('StdDev: ' + str(df[col_name].std()))
        print('Distinct: ' + str(df[col_name].nunique()))
        print('Sum: ' + str(df[col_name].sum()))
        print('Missing: ' + str(df[col_name].isnull().sum()))
        print('Missing%: ' + str(df[col_name].isnull().sum() / df.shape[0]))
        print('Skewness: ' + str(df[col_name].skew()))
        print('Kurtosis: ' + str(df[col_name].kurtosis()))

        Q1 = df[col_name].quantile(q=0.25)
        Q3 = df[col_name].quantile(q=0.75)
        IQR = Q3 - Q1
        upper_bound = 1.5 * IQR + Q3
        lower_bound = Q1 - 1.5 * IQR
        outliers = len(df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)])

        print('Outliers: ' + str(outliers))
        print('25%: ' + str(Q1))
        print('75%: ' + str(Q3))
        print('IQR: ' + str(IQR))
        print('Down: ' + str(lower_bound))
        print('Up: ' + str(upper_bound))
        print('-------------------------------------')
        fig, ax = plt.subplots(2, 1)
        sns.histplot(data=df, x=col_name, kde=True, ax=ax[0])
        sns.boxplot(x=df[col_name], ax=ax[1])

        if return_result:
            result = dict()
            result['Type'] = str(df[col_name].dtype)
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
            result['Q1'] = Q1
            result['Q3'] = Q3
            result['IQR'] = IQR
            result['Down'] = lower_bound
            result['Up'] = upper_bound
            return result
