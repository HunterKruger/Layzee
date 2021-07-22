import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# test passed
class FeatureDrift:
    """
    Visualize and detect feature drift.
    """

    def __init__(self):
        pass

    @staticmethod
    def adversarial_detection(df_train, df_test, target_col=None, roc_tolerance=0.005, random_state=1234,
                              return_result=False):
        """
        By training an Adversarial Classifier to determine whether there is feature drift.
        The most important feature will be dropped and the Adversarial Classifier will be retrained at each iteration
        until auc_roc reduces into the specified tolerance range.
        Reference: https://zhuanlan.zhihu.com/p/349432455
        :param df_train: training set; remember to drop id features!
        :param df_test: test set; remember to drop id features!
        :param target_col: name of the target column, which will be dropped from df_train and df_test if specified
        :param roc_tolerance: tolerance of auroc
            eg: 0.005 -> raise feature drift warning when auroc > 0.505 or < 0.495
        :param random_state: random state seed
        :param return_result: return df_train & df_test without deleted features if True

        """
        df_train_ = df_train.copy()
        df_test_ = df_test.copy()

        print('roc safe range: [' + str(round((0.5 - roc_tolerance), 4)) + ', ' +
              str(round((0.5 + roc_tolerance), 4)) + ']')

        model = RandomForestClassifier(random_state=random_state)
        df_train_['fake_label'] = 0
        df_test_['fake_label'] = 1

        if target_col is not None:
            df_train_.drop(target_col, axis=1, inplace=True)
            df_test_.drop(target_col, axis=1, inplace=True)

        df_all = pd.concat([df_train_, df_test_], axis=0)
        X = df_all.drop('fake_label', axis=1)
        y = df_all['fake_label']
        # print(y.value_counts())

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, train_size=0.75, random_state=random_state)

        X_train, X_test = None, None
        y_train, y_test = None, None

        for train_index, test_index in sss.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            # print("Overlap: "+str(len(train_index.intersection(test_index))))
            X_train, X_test = X.loc[X.index.intersection(train_index)], X.loc[X.index.intersection(test_index)]
            y_train, y_test = y.loc[y.index.intersection(train_index)], y.loc[y.index.intersection(test_index)]
        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        # print(y_train.value_counts())
        # print(y_test.value_counts())

        features_to_drop = []

        hp = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}
        optimizer = GridSearchCV(estimator=model, param_grid=hp, n_jobs=-1, cv=3, scoring='roc_auc', refit=True)

        while True:
            optimizer.fit(X_train, y_train)
            best_model = optimizer.best_estimator_
            y_score = best_model.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, y_score)

            if roc < 0.5 - roc_tolerance or roc > 0.5 + roc_tolerance:
                pack = sorted(zip(X_train.columns.tolist(), best_model.feature_importances_.tolist()),
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
                if return_result:
                    return df_train_, df_test_, features_to_drop
                else:
                    break

    @staticmethod
    def categorical_detection(df_train, df_test, col_name, top_n=None, plot_size_x=12, plot_size_y=12,
                              return_result=False):
        """
        Compare a numerical feature in 2 dataframes.
        :param df_train: training set
        :param df_test: test set
        :param col_name: column name of this categorical feature
        :param top_n: top n classes to be counted and plotted
        :param plot_size_x: plot size in x-axis
        :param plot_size_y: plot size in y-axis
        :param return_result: return result if True
        :return: all plotted stats
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

        # print('-----------------df_train summary--------------------------')
        # for k, v in result1.items():
        #     print(str(k) + ': ' + str(v))
        # print('-----------------df_test summary--------------------------')
        # for k, v in result2.items():
        #     print(str(k) + ': ' + str(v))

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
            df_train[col_name].value_counts(normalize=True).nlargest(top_n).cumsum()],
            axis=1)

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

        fig = plt.figure(figsize=(plot_size_x, plot_size_y))
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        plt.pie(count_table_info1['Count'][:top_n], labels=count_table_info1[col_name][:top_n], autopct='%1.2f%%')
        plt.title(col_name + ' in df1')
        ax2 = plt.subplot2grid((1, 2), (0, 1))
        plt.pie(count_table_info2['Count'][:top_n], labels=count_table_info2[col_name][:top_n], autopct='%1.2f%%')
        plt.title(col_name + ' in df2')

        if return_result:
            return result_df, count_table_info1, count_table_info2

    @staticmethod
    def numerical_detection(df_train, df_test, col_name, plot_size_x=10, plot_size_y=5, return_result=False):
        """
        Compare a numerical feature in 2 dataframes.
        :param df_train: training set
        :param df_test: test set
        :param col_name: column name of this categorical feature
        :param plot_size_x: plot size in x-axis
        :param plot_size_y: plot size in y-axis
        :param return_result: return result if True
        :return: all plotted stats
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

        plot_min = min(df_train[col_name].min(), df_test[col_name].min())
        plot_max = max(df_train[col_name].max(), df_test[col_name].max())

        df_plot_1 = df_train[col_name].to_frame()
        df_plot_1['label'] = 'df_train'
        df_plot_2 = df_test[col_name].to_frame()
        df_plot_2['label'] = 'df_test'
        df_plot = pd.concat([df_plot_1, df_plot_2], axis=0)
        # print(df_plot)
        # sns.displot(data=df_plot, x=col_name, hue="label", kind='kde')

        fig, ax = plt.subplots(3, 1, figsize=(plot_size_x, plot_size_y))
        ax[0].set_xlim(plot_min, plot_max)
        ax[1].set_xlim(plot_min, plot_max)
        sns.histplot(data=df_train, x=col_name, kde=True, ax=ax[0])
        sns.histplot(data=df_test, x=col_name, kde=True, ax=ax[1])
        sns.boxplot(data=df_plot, x=col_name, y='label', ax=ax[2])

        if return_result:
            return result_df
