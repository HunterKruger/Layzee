import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp, chi2_contingency
from layzee.splitter_sampler import split_df

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# test passed
def adversarial_detection(df_train, df_test, roc_tolerance=0.05, random_state=1234):
    """
    By training an Adversarial Classifier to determine whether there is feature drift.
    The most important feature will be dropped and the Adversarial Classifier will be retrained at each iteration
    until auc_roc reduces into a specified tolerance range.
    Reference: https://zhuanlan.zhihu.com/p/349432455
    :param df_train: training set; remember to drop id features and target!
    :param df_test: test set; remember to drop id features and target!
    :param roc_tolerance: tolerance of auroc
        eg: 0.005 -> raise feature drift warning when  0.495 < auroc < 0.505
    :param random_state: random state seed
    """
    df_train_ = df_train.copy()
    df_test_ = df_test.copy()

    print(
        'roc safe range: [' + str(round((0.5 - roc_tolerance), 4)) + ', ' + str(round((0.5 + roc_tolerance), 4)) + ']')

    df_train_['fake_label'] = 0
    df_test_['fake_label'] = 1

    df_all = pd.concat([df_train_, df_test_], axis=0)
    df_all = df_all.sample(frac=1)
    df_all.reset_index(inplace=True)
    df_all.drop('index', axis=1, inplace=True)

    X_train, X_test, y_train, y_test = split_df(df_all, test_ratio=0.5, target='fake_label', random_state=random_state)

    features_to_drop = []

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)

    while True:
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_score)

        if roc < 0.5 - roc_tolerance or roc > 0.5 + roc_tolerance:
            pack = sorted(zip(X_train.columns.tolist(), model.feature_importances_.tolist()),
                          key=lambda tup: tup[1], reverse=True)
            data, idx = zip(*pack)
            first_feature = data[0]
            print('Feature drift warning (roc = ' + str(round(roc, 4)) + '): drop ' + first_feature)
            X_train.drop(first_feature, axis=1, inplace=True)
            X_test.drop(first_feature, axis=1, inplace=True)
            features_to_drop.append(first_feature)
        else:
            print('No feature drift detected  (roc = ' + str(round(roc, 4)) + ')')
            print('The following features have been dropped:')
            print(features_to_drop)
            return features_to_drop


def categorical_detection(df_train, df_test, col_name, top_n=None, plot_size=(12, 12), return_result=False,
                          file_name=None, adv_drift=False):
    """
    Compare a numerical feature in 2 dataframes.
    :param df_train: training set
    :param df_test: test set
    :param col_name: column name of this categorical feature
    :param top_n: top n classes in df_train to be counted and plotted
    :param plot_size: plot size (x, y)
    :param return_result: return result if True
    :param file_name: path + filename to save the plot
    :param adv_drift: drift detected by adversarial validation method
    """

    result1 = dict()
    result1['Type'] = str(df_train[col_name].dtype)
    result1['Rows'] = df_train.shape[0]
    result1['Distinct'] = df_train[col_name].nunique()
    result1['Missing'] = df_train[col_name].isnull().sum()
    result1['Missing%'] = df_train[col_name].isnull().sum() / df_train.shape[0]

    result2 = dict()
    result2['Type'] = str(df_test[col_name].dtype)
    result2['Rows'] = df_test.shape[0]
    result2['Distinct'] = df_test[col_name].nunique()
    result2['Missing'] = df_test[col_name].isnull().sum()
    result2['Missing%'] = df_test[col_name].isnull().sum() / df_test.shape[0]

    result_df = pd.DataFrame(index=result1.keys())
    result_df['df_train'] = result1.values()
    result_df['df_test'] = result2.values()
    print(result_df)

    if top_n is None:
        top_n = len(df_train[col_name].value_counts())

    print('-----------------df_train top ' + str(top_n) + ' values:--------------------')
    count_table_info1 = pd.concat([
        df_train[col_name].value_counts().nlargest(top_n),
        df_train[col_name].value_counts(normalize=True).nlargest(top_n),
        df_train[col_name].value_counts(normalize=True).nlargest(top_n).cumsum()], axis=1)

    count_table_info1.columns = ['Count', '%', 'Cum.%']
    count_table_info1 = count_table_info1.sort_values(by='Count', ascending=False)
    count_table_info1.reset_index(inplace=True)
    count_table_info1.columns = [col_name, 'Count', '%', 'Cum.%']
    print(count_table_info1)

    if top_n is None:
        top_n = len(df_test[col_name].value_counts())

    print('-----------------df_test top ' + str(top_n) + ' values:--------------------')
    count_table_info2 = pd.concat([
        df_test[col_name].value_counts().nlargest(top_n),
        df_test[col_name].value_counts(normalize=True).nlargest(top_n),
        df_test[col_name].value_counts(normalize=True).nlargest(top_n).cumsum()], axis=1)

    count_table_info2.columns = ['Count', '%', 'Cum.%']
    count_table_info2 = count_table_info2.sort_values(by='Count', ascending=False)
    count_table_info2.reset_index(inplace=True)
    count_table_info2.columns = [col_name, 'Count', '%', 'Cum.%']
    print(count_table_info2)

    print('-----------------Chi-square test--------------------')
    cross_table = pd.crosstab(df_train[col_name], df_test[col_name], margins=False)
    sig_lv = 0.05
    chi_value, p_value, deg_free, _ = chi2_contingency(cross_table)
    print('Chi-square statistic: ', chi_value)
    print('Degrees of freedom: ', deg_free)
    print('Significance level: ', sig_lv)
    print('p-value: ', p_value)
    has_drift = False
    if p_value >= sig_lv:
        print('<' + col_name + ' in these 2 datasets are independent> cannot be rejected.')
        print('Drift detected!')
        has_drift = True
    else:
        print('<' + col_name + ' in these 2 datasets are independent> can be rejected.')
        print('No drift.')

    fig = plt.figure(figsize=plot_size)
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    plt.pie(count_table_info1['Count'][:top_n], labels=count_table_info1[col_name][:top_n], autopct='%1.2f%%')
    plt.title(col_name + ' in df1')
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    plt.pie(count_table_info2['Count'][:top_n], labels=count_table_info2[col_name][:top_n], autopct='%1.2f%%')
    plt.title(col_name + ' in df2')

    if (file_name is not None) and (has_drift or adv_drift):
        fig.savefig(file_name)

    if return_result:
        return has_drift, result_df, count_table_info1, count_table_info2


def numerical_detection(df_train, df_test, col_name, plot_size=(10, 5), return_result=False, file_name=None,
                        adv_drift=False):
    """
    Compare a numerical feature in 2 dataframes.
    :param df_train: training set
    :param df_test: test set
    :param col_name: column name of this categorical feature
    :param plot_size: plot size (x,y)
    :param return_result: return result if True
    :param file_name: path + filename to save the plot
    :param adv_drift: drift detected by adversarial validation method
    """

    q1_1 = df_train[col_name].quantile(q=0.25)
    q3_1 = df_train[col_name].quantile(q=0.75)
    iqr_1 = q3_1 - q1_1
    upper_bound_1 = 1.5 * iqr_1 + q3_1
    lower_bound_1 = q1_1 - 1.5 * iqr_1
    outliers_1 = len(df_train[(df_train[col_name] < lower_bound_1) | (df_train[col_name] > upper_bound_1)])

    result1 = dict()
    result1['Type'] = df_train[col_name].dtype
    result1['Rows'] = df_train.shape[0]
    result1['Min'] = df_train[col_name].min()
    result1['Max'] = df_train[col_name].max()
    result1['Mean'] = df_train[col_name].mean()
    result1['Median'] = df_train[col_name].median()
    result1['Mode'] = df_train[col_name].value_counts().index[0]
    result1['StdDev'] = df_train[col_name].std()
    result1['Distinct'] = df_train[col_name].nunique()
    result1['Sum'] = df_train[col_name].sum()
    result1['Missing'] = df_train[col_name].isnull().sum()
    result1['Missing%'] = df_train[col_name].isnull().sum() / df_train.shape[0]
    result1['Skewness'] = df_train[col_name].skew()
    result1['Kurtosis'] = df_train[col_name].kurtosis()
    result1['Outliers'] = outliers_1
    result1['Q1'] = q1_1
    result1['Q3'] = q3_1
    result1['IQR'] = iqr_1
    result1['Down'] = lower_bound_1
    result1['Up'] = upper_bound_1

    q1_2 = df_test[col_name].quantile(q=0.25)
    q3_2 = df_test[col_name].quantile(q=0.75)
    iqr_2 = q3_2 - q1_2
    upper_bound_2 = 1.5 * iqr_2 + q3_2
    lower_bound_2 = q1_2 - 1.5 * iqr_2
    outliers_2 = len(df_test[(df_test[col_name] < lower_bound_2) | (df_test[col_name] > upper_bound_2)])

    result2 = dict()
    result2['Type'] = df_test[col_name].dtype
    result2['Rows'] = df_test.shape[0]
    result2['Min'] = df_test[col_name].min()
    result2['Max'] = df_test[col_name].max()
    result2['Mean'] = df_test[col_name].mean()
    result2['Median'] = df_test[col_name].median()
    result2['Mode'] = df_test[col_name].value_counts().index[0]
    result2['StdDev'] = df_test[col_name].std()
    result2['Distinct'] = df_test[col_name].nunique()
    result2['Sum'] = df_test[col_name].sum()
    result2['Missing'] = df_test[col_name].isnull().sum()
    result2['Missing%'] = df_test[col_name].isnull().sum() / df_test.shape[0]
    result2['Skewness'] = df_test[col_name].skew()
    result2['Kurtosis'] = df_test[col_name].kurtosis()
    result2['Outliers'] = outliers_2
    result2['Q1'] = q1_2
    result2['Q3'] = q3_2
    result2['IQR'] = iqr_2
    result2['Down'] = lower_bound_2
    result2['Up'] = upper_bound_2

    result_df = pd.DataFrame(index=result1.keys())
    result_df['df_train'] = result1.values()
    result_df['df_test'] = result2.values()

    print(result_df)
    print()

    print('Kolmogorov-Smirnov test:')
    ks_result = ks_2samp(df_train[col_name], df_test[col_name])
    sig_lv = 0.05
    print('KS statistic: ', ks_result.statistic)
    print('Significance level: ', sig_lv)
    print('p-value: ', ks_result.pvalue)
    has_drift = False
    if ks_result.pvalue > sig_lv:
        print('<These 2 distributions follows the same distribution> cannot be rejected.')
        print('No drift.')
    else:
        print('<These 2 distributions follows the same distribution> can be rejected.')
        print('Drift')
        has_drift = True

    plot_min = min(df_train[col_name].min(), df_test[col_name].min())
    plot_max = max(df_train[col_name].max(), df_test[col_name].max())

    df_plot_1 = df_train[col_name].to_frame()
    df_plot_1['label'] = 'df_train'
    df_plot_2 = df_test[col_name].to_frame()
    df_plot_2['label'] = 'df_test'
    df_plot = pd.concat([df_plot_1, df_plot_2], axis=0)

    plt.figure()
    fig, ax = plt.subplots(3, 1, figsize=plot_size)
    ax[0].set_xlim(plot_min, plot_max)
    ax[1].set_xlim(plot_min, plot_max)
    sns.histplot(data=df_train, x=col_name, kde=True, ax=ax[0])
    sns.histplot(data=df_test, x=col_name, kde=True, ax=ax[1])
    sns.boxplot(data=df_plot, x=col_name, y='label', ax=ax[2])

    if (file_name is not None) and (has_drift or adv_drift):
        fig.savefig(file_name)

    if return_result:
        return has_drift, result_df


def auto_detection(df_train, df_test, col_name, return_result=False, file_name=None, adv_drift=False):
    if df_train[col_name].dtype == 'object':
        has_drift, result_df, _, _ = categorical_detection(
            df_train=df_train, df_test=df_test, col_name=col_name,
            return_result=return_result, file_name=file_name, adv_drift=adv_drift
        )
    else:
        has_drift, result_df = numerical_detection(
            df_train=df_train, df_test=df_test, col_name=col_name,
            return_result=return_result, file_name=file_name, adv_drift=adv_drift
        )
    return has_drift, result_df
