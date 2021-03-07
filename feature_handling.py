import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer


class FeatureHandling:

    def __init__(self, df1, df2=None):
        self.df1 = df1
        self.df2 = df2

    def imputer(self, col, method='mode', groupby=None):
        """
        Impute missing values in a column
        :param col: column name
        :param method: choose the method of imputing, a customized value is allowed
        :param groupby -> str or list of str:
                group by one or several categorical columns to calculate the values for imputing
        """

        if groupby is None:
            if method == 'mean':
                temp = self.df1[col].mean()
            elif method == 'median':
                temp = self.df1[col].median()
            elif method == 'mode':
                temp = self.df1[col].mode()[0]
            else:
                temp = method

            self.df1[col].fillna(temp, inplace=True)
            if self.df2 is not None:
                self.df2[col].fillna(temp, inplace=True)
        else:
            if method == 'mean':
                self.df1[col] = self.df1.groupby(groupby)[col].apply(lambda x: x.fillna(x.mean()))
            elif method == 'median':
                self.df1[col] = self.df1.groupby(groupby)[col].apply(lambda x: x.fillna(x.median()))
            elif method == 'mode':
                self.df1[col] = self.df1.groupby(groupby)[col].apply(lambda x: x.fillna(x.mode()[0]))
            else:
                print("Please set 'groupby' to None!")

            ### todo: transform on df2

    def imputers(self, method_dict=dict()):
        """
        Impute missing values in several columns at a time
        :param method_dict: a dict indicating impute method for each column
                            eg: {col1:'mode', col2:'mean', col3:'other'}
        """
        for col in method_dict.keys():
            self.imputer(col, method_dict[col])

    # test passed
    def keep_top_n(self, col, N, include_nan=False, result=np.nan):
        """
        Merge long tail to a specified value
        :param col: column name, should be categorical
        :param N: keep top N class if N is integer(N>=1);
                  keep classes whose percentage is higher then N if N is decimal(0<N<1)
        :param include_nan: include nan when counting top N classes
        :param result: the value to replace long tail
        """
        if include_nan is True and N >= 1:
            top_n = self.df1[col].value_counts(dropna=False).index.tolist()[:N]
        if include_nan is False and N >= 1:
            top_n = self.df1[col].value_counts().index.tolist()[:N]
        if include_nan is True and N < 1:
            rank = self.df1[col].value_counts(dropna=False, normalize=True)
            real_n = sum(i > N for i in rank.values.tolist())
            top_n = self.df1[col].value_counts(dropna=False).index.tolist()[:real_n]
        if include_nan is False and N < 1:
            rank = self.df1[col].value_counts(normalize=True)
            real_n = sum(i > N for i in rank.values.tolist())
            top_n = self.df1[col].value_counts().index.tolist()[:real_n]

        def f(x):
            if x not in top_n:
                return result
            else:
                return x

        self.df1[col] = self.df1[col].apply(lambda x: f(x))
        if self.df2 is not None:
            self.df2[col] = self.df2[col].apply(lambda x: f(x))

    # test passed
    def handle_outlier(self, col, drop_row=False):
        """
        Handle outliers in a numerical column.
        Should not contain missing values, or they will be imputed by median.
        :param col: column name, must be numerical
        :param drop_row:
                choose 'False' to replace outliers by np.nan
                choose 'True' to drop rows with outliers
        """

        self.imputer(col, 'median')

        q1 = self.df1[col].quantile(q=0.25)
        q3 = self.df1[col].quantile(q=0.75)
        iqr = q3 - q1
        upper_bound = 1.5 * iqr + q3
        lower_bound = q1 - 1.5 * iqr

        if not drop_row:
            self.df1.loc[(self.df1[col] < lower_bound) | (self.df1[col] > upper_bound), col] = np.nan
            if self.df2 is not None:
                self.df2.loc[(self.df1[col] < lower_bound) | (self.df1[col] > upper_bound), col] = np.nan
        else:
            self.df1 = self.df1[(self.df1[col] <= upper_bound) & (self.df1[col] >= lower_bound)]
            if self.df2 is not None:
                self.df2 = self.df2[(self.df2[col] <= upper_bound) & (self.df2[col] >= lower_bound)]

    def outlier_encoder(self, col, standard_scaling=True, drop=True, method='cut'):
        """
        Encode outliers in this column by 3 new features.
        Should not contain missing values, or they will be imputed by median.
        :param col: column name, must be numerical
        :param standard_scaling: scaling by std and mean after encoding
        :param drop: drop the origin feature
        :param method:
            'cut': outliers replaced by upper and lower bounds.
            'mean': outliers replaced by mean.
            'median': outliers replaced by median.
        """

        self.imputer(col, 'mean')

        q1 = self.df1[col].quantile(q=0.25)
        q3 = self.df1[col].quantile(q=0.75)
        iqr = q3 - q1
        upper_bound = 1.5 * iqr + q3
        lower_bound = q1 - 1.5 * iqr
        std = self.df1[col].std()
        mean = self.df1[col].mean()
        median = self.df1[col].median()

        def extract_lower_bound(x):
            if x < lower_bound:
                return 1
            else:
                return 0

        def extract_upper_bound(x):
            if x > upper_bound:
                return 1
            else:
                return 0

        def cut_outliers(x):
            if x > upper_bound:
                return upper_bound
            elif x < lower_bound:
                return lower_bound
            return x

        def mean_outliers(x):
            if x > upper_bound or x < lower_bound:
                return mean
            else:
                return x

        def median_outliers(x):
            if x > upper_bound or x < lower_bound:
                return median
            else:
                return x

        def mean_std_scaling(x):
            return (x - mean) / std

        self.df1[col + '_lower'] = self.df1[col].apply(lambda x: extract_lower_bound(x))
        self.df1[col + '_upper'] = self.df1[col].apply(lambda x: extract_upper_bound(x))
        if method == 'cut':
            self.df1[col + '_inner'] = self.df1[col].apply(lambda x: cut_outliers(x))
        elif method == 'mean':
            self.df1[col + '_inner'] = self.df1[col].apply(lambda x: mean_outliers(x))
        elif method == 'median':
            self.df1[col + '_inner'] = self.df1[col].apply(lambda x: median_outliers(x))
        if standard_scaling:
            self.df1[col + '_inner'] = self.df1[col].apply(lambda x: mean_std_scaling(x))

        if self.df2 is not None:
            self.df2[col + '_lower'] = self.df2[col].apply(lambda x: extract_lower_bound(x))
            self.df2[col + '_upper'] = self.df2[col].apply(lambda x: extract_upper_bound(x))
            if method == 'cut':
                self.df2[col + '_inner'] = self.df2[col].apply(lambda x: cut_outliers(x))
            elif method == 'mean':
                self.df2[col + '_inner'] = self.df2[col].apply(lambda x: mean_outliers(x))
            elif method == 'median':
                self.df2[col + '_inner'] = self.df2[col].apply(lambda x: median_outliers(x))
            if standard_scaling:
                self.df2[col] = self.df2[col].apply(lambda x: mean_std_scaling(x))

        if drop:
            self.drop(col)

    @staticmethod
    def get_skewness(df, col):
        # exclude outer 3% data and 0
        q_015, q_985 = df[col].quantile([0.015, 0.985])
        return df[col][(col >= q_015) & (col <= q_985) & (col != 0)].skew()

    def handle_skewness(self, col, drop=True):
        """
        log(1+x) transformation if col>0 and -1<skewness<1
        Only do this for one DataFrame: df1
        :param col: column name, must be numerical
        :param drop: drop the origin feature
        """
        skewness = self.get_skewness(self.df1, col)
        print('Skewness: ' + str(skewness))

        if skewness <= 1 or skewness >= -1:
            print('No need of transformation')
        else:
            min_value = self.df1[col].min()
            if min_value < 0:
                self.df1[col + '_log1x'] = self.df1[col] + min_value
            self.df1[col + '_log1x'] = np.log1p(self.df1[col])

            if self.df2 is not None:
                if self.df2.min() < min_value:
                    print("Failed to handle skewness in df2")
                    pass
                elif min_value < 0:
                    self.df2[col + '_log1x'] = self.df2[col] + min_value
                self.df2[col + '_log1x'] = np.log1p(self.df2[col])
            print('Skewness: ' + self.get_skewness(self.df1, col + '_log1x'))
            if drop:
                self.drop(col)

    # test passed
    def binning(self, col, threshold, reverse=False, drop=True):
        """
        Binarize a column based on a threshold
        :param col: column name
        :param threshold: a threshold to split data
        :param reverse:
                False: output = 1 when value is larger than threshold
                True: output = 1 when value is smaller than threshold
        :param drop: drop the origin column
        """

        def bin(x):
            if np.isnan(x):
                return x
            elif x < threshold:
                return 0
            else:
                return 1

        def bin_reverse(x):
            if np.isnan(x):
                return x
            elif x < threshold:
                return np.nan
            else:
                return 0

        if not reverse:
            self.df1[col + '_binned'] = self.df1[col].apply(lambda x: bin(x))
            if self.df2 is not None:
                self.df2[col + '_binned'] = self.df2[col].apply(lambda x: bin(x))
        else:
            self.df1[col + '_binned'] = self.df1[col].apply(lambda x: bin_reverse(x))
            if self.df2 is not None:
                self.df2[col + '_binned'] = self.df2[col].apply(lambda x: bin_reverse(x))

        if drop:
            self.drop(col)

    # test passed
    def drop(self, col):
        self.df1.drop(col, axis=1, inplace=True)
        if self.df2 is not None:
            self.df2.drop(col, axis=1, inplace=True)

    # test passed
    def indicator(self, col, target=np.nan, reverse=False, drop=True):
        """
        Create an indicator column to indicate the presence of a specific value
        :param col: column name
        :param target: the specific value for creating indicator
        :param reverse:
                False: value == target, then gives 1
                True: value == target, then gives 0
        :param drop: drop the origin column
        """

        def idc(x):
            if np.isnan(target):
                return 1 if np.isnan(x) else 0
            else:
                return 1 if x == target else 0

        def idc_reverse(x):
            if np.isnan(target):
                return 1 if np.isnan(x) else 0
            else:
                return 0 if x == target else 1

        if not reverse:
            self.df1[col + '_idc'] = self.df1[col].apply(lambda x: idc(x))
            if self.df2 is not None:
                self.df2[col + '_idc'] = self.df2[col].apply(lambda x: idc(x))
        else:
            self.df1[col + '_idc'] = self.df1[col].apply(lambda x: idc_reverse(x))
            if self.df2 is not None:
                self.df2[col + '_idc'] = self.df2[col].apply(lambda x: idc_reverse(x))

        if drop:
            self.drop(col)

    # test passed
    def cut(self, col, bins, labels=False, drop=True):
        """
        Cut a column based on number of bins or customized bin edges.
        Should not contain missing values, or they will be imputed by median.
        :param col: column name
        :param bins:
                number of bins: set number of bins, each bin has the same range size
                list of bin edges: set bin edges manually
        :param labels: list of label, len(labels) == bins or len(bin)-1
        :param drop: drop the origin column
        """

        self.imputer(col, 'median')
        self.df1[col + '_cut'], bins_df1 = pd.cut(self.df1[col], bins=bins, labels=labels, retbins=True)
        if self.df2 is not None:
            self.df2[col + '_cut'] = pd.cut(self.df2[col], bins=bins_df1, labels=labels)
        if drop:
            self.drop(col)

    # test passed
    def qcut(self, col, q, labels=False, drop=True):
        """
        Cut a column based on quantiles
        :param col: column name
        :param q: number of bins, each bin gets same nb of samples
        :param labels: list of label, len(labels) == q
        :param drop: drop the origin column
        """
        self.df1[col + '_qcut'], bins_df1 = pd.qcut(self.df1[col], q=q, labels=labels, retbins=True)
        if self.df2 is not None:
            self.df2[col + '_qcut'] = pd.cut(self.df2[col], bins=bins_df1, labels=labels)
        if drop:
            self.drop(col)

    def general_encoder(self, num_cols, cat_cols):
        """
        Encode categorical features by one-hot, numerical features by mean-std scaling
        No dropping the first column after one-hot, so this function is not suitable for linear models
        :param num_cols: list of numerical columns to be encoded
        :param cat_cols: list of categorical columns to be encoded
        :return:
        """
        df1_rest = list(set(self.df1.columns.tolist()) - set(num_cols) - set(cat_cols))
        dv = DictVectorizer(sparse=False)
        cls_cat1 = dv.fit_transform(self.df1[cat_cols].to_dict(orient='record'))
        df1_cat = pd.DataFrame(data=cls_cat1, columns=dv.feature_names_)

        ss = StandardScaler()
        cls_num1 = ss.fit_transform(self.df1[num_cols])
        df1_num = pd.DataFrame(cls_num1, columns=num_cols)

        df1_encoded = pd.concat([df1_num, df1_cat, df1_rest], axis=1)

        if self.df2 is None:
            return df1_encoded
        else:
            df2_rest = self.df2.columns.tolist() - num_cols - cat_cols
            cls_cat2 = dv.transform(self.df2[cat_cols].to_dict(orient='record'))
            df2_cat = pd.DataFrame(data=cls_cat2, columns=dv.feature_names_)
            cls_num2 = ss.transform(self.df2[num_cols])
            df2_num = pd.DataFrame(cls_num2, columns=num_cols)
            df2_encoded = pd.concat([df2_num, df2_cat, df2_rest], axis=1)
            return df1_encoded, df2_encoded
