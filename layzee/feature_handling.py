import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder


class FeatureHandling:
    """
    Several feature handling methods.
    This class supports only one dataframe, it will be fit then transformed.
    """

    def __init__(self, df, drop_origin=True):
        """
        :param df: a dataframe to be fit then transformed
        :param drop_origin: drop origin feature if new ones are created
        """
        self.df = df.copy()
        self.drop_origin = drop_origin

    def imputer(self, col, method='mode', groupby=None):
        """
        Impute missing values in a column
        :param groupby: -> str or list of str:
                group by one or several categorical columns to calculate the values for imputing
        :param col: column name
        :param method: choose the method of imputing, a customized value is allowed
        """

        if groupby is None:
            if method == 'mean':
                temp = self.df[col].mean()
            elif method == 'median':
                temp = self.df[col].median()
            elif method == 'mode':
                temp = self.df[col].mode()[0]
            else:
                temp = method

            self.df[col].fillna(temp, inplace=True)

        else:
            if method == 'mean':
                self.df[col] = self.df.groupby(groupby)[col].apply(lambda x: x.fillna(x.mean()))
            elif method == 'median':
                self.df[col] = self.df.groupby(groupby)[col].apply(lambda x: x.fillna(x.median()))
            elif method == 'mode':
                self.df[col] = self.df.groupby(groupby)[col].apply(lambda x: x.fillna(x.mode()[0]))
            else:
                print("Please set 'groupby' to None!")

            ### todo: transform on df2

    def manual_imputers(self, method_dict):
        """
        Impute missing values in several columns at a time, must specify the imputing method for each column
        :param method_dict: a dictionary indicating impute method for each column
                            eg: {col1:'mode', col2:'mean', col3:'other'}
        """
        for col in method_dict.keys():
            self.imputer(col, method_dict[col])

    def auto_imputers(self, cat_method='mode', num_method='median'):
        """
        Impute missing values in several columns at a time, automatically detect categorical and numerical features
        :param cat_method: imputing method for categorical features
        :param num_method: imputing method for numerical features
        """
        for col in self.df.select_dtypes('object').columns.tolist():
            self.imputer(col, cat_method)
        for col in self.df.select_dtypes('number').columns.tolist():
            self.imputer(col, num_method)

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
        top_n = []
        if include_nan is True and N >= 1:
            top_n = self.df[col].value_counts(dropna=False).index.tolist()[:N]
        if include_nan is False and N >= 1:
            top_n = self.df[col].value_counts().index.tolist()[:N]
        if include_nan is True and N < 1:
            rank = self.df[col].value_counts(dropna=False, normalize=True)
            real_n = sum(i > N for i in rank.values.tolist())
            top_n = self.df[col].value_counts(dropna=False).index.tolist()[:real_n]
        if include_nan is False and N < 1:
            rank = self.df[col].value_counts(normalize=True)
            real_n = sum(i > N for i in rank.values.tolist())
            top_n = self.df[col].value_counts().index.tolist()[:real_n]
        self.df[col] = self.df[col].apply(lambda x: x if x in top_n else result)
        return top_n

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

        q1 = self.df[col].quantile(q=0.25)
        q3 = self.df[col].quantile(q=0.75)
        iqr = q3 - q1
        upper_bound = 1.5 * iqr + q3
        lower_bound = q1 - 1.5 * iqr

        if not drop_row:
            self.df.loc[(self.df[col] < lower_bound) | (self.df[col] > upper_bound), col] = np.nan
        else:
            self.df = self.df[(self.df[col] <= upper_bound) & (self.df[col] >= lower_bound)]

        return upper_bound, lower_bound

    def outlier_encoder(self, col, standard_scaling=True, method='cut'):
        """
        Encode outliers in this column by 3 new features.
        Should not contain missing values, or they will be imputed by median.
        :param col: column name, must be numerical
        :param standard_scaling: scaling by std and mean after encoding
        :param method:
            'cut': outliers replaced by upper and lower bounds.
            'mean': outliers replaced by mean.
            'median': outliers replaced by median.
        """

        self.imputer(col, 'mean')

        q1 = self.df[col].quantile(q=0.25)
        q3 = self.df[col].quantile(q=0.75)
        iqr = q3 - q1
        upper_bound = 1.5 * iqr + q3
        lower_bound = q1 - 1.5 * iqr
        std = self.df[col].std()
        mean = self.df[col].mean()
        median = self.df[col].median()

        def extract_lower_bound(x):
            return 1 if x < lower_bound else 0

        def extract_upper_bound(x):
            return 1 if x > upper_bound else 0

        def cut_outliers(x):
            if x > upper_bound:
                return upper_bound
            elif x < lower_bound:
                return lower_bound
            return x

        def mean_outliers(x):
            return mean if x > upper_bound or x < lower_bound else x

        def median_outliers(x):
            return median if x > upper_bound or x < lower_bound else x

        def mean_std_scaling(x):
            return (x - mean) / std

        self.df[col + '_lower'] = self.df[col].apply(lambda x: extract_lower_bound(x))
        self.df[col + '_upper'] = self.df[col].apply(lambda x: extract_upper_bound(x))
        if method == 'cut':
            self.df[col + '_inner'] = self.df[col].apply(lambda x: cut_outliers(x))
        elif method == 'mean':
            self.df[col + '_inner'] = self.df[col].apply(lambda x: mean_outliers(x))
        elif method == 'median':
            self.df[col + '_inner'] = self.df[col].apply(lambda x: median_outliers(x))
        if standard_scaling:
            self.df[col + '_inner'] = self.df[col].apply(lambda x: mean_std_scaling(x))

        if self.drop_origin:
            self.drop(col)

    def get_skewness(self, col):
        # exclude outer 3% data
        q_015, q_985 = self.df[col].quantile([0.015, 0.985])
        return self.df[col][(col >= q_015) & (col <= q_985)].skew()

    def handle_skewness(self, col):
        """
        log(1+x) transformation if col>0 and -1<skewness<1
        Only do this for one DataFrame: df1
        :param col: column name, must be numerical
        """
        skewness = self.get_skewness(col)
        print('Skewness: ' + str(skewness))

        if skewness <= 1 or skewness >= -1:
            print('No need of transformation')
        else:
            min_value = self.df[col].min()
            if min_value < 0:
                # move within x-axis
                self.df[col + '_log1x'] = self.df[col] + min_value
            # log(1+x) transformation
            self.df[col + '_log1x'] = np.log1p(self.df[col])

            print('Skewness: ' + self.get_skewness(self.df, col + '_log1x'))

            if self.drop_origin:
                self.drop(col)

    # test passed
    def binning(self, col, threshold, reverse=False):
        """
        Binarize a column based on a threshold
        :param col: column name
        :param threshold: a threshold to split data
        :param reverse:
                False: output = 1 when value is larger than threshold
                True: output = 1 when value is smaller than threshold
        """

        def bin_(x):
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
            self.df[col + '_binned'] = self.df[col].apply(lambda x: bin_(x))
        else:
            self.df[col + '_binned'] = self.df[col].apply(lambda x: bin_reverse(x))

        if self.drop_origin:
            self.drop(col)

    # test passed
    def drop(self, col):
        self.df.drop(col, axis=1, inplace=True)

    # test passed
    def indicator(self, col, target=np.nan, reverse=False):
        """
        Create an indicator column to indicate the presence of a specific value
        :param col: column name
        :param target: the specific value for creating indicator
        :param reverse:
                False: value == target, then gives 1
                True: value == target, then gives 0
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
            self.df[col + '_idc'] = self.df[col].apply(lambda x: idc(x))
        else:
            self.df[col + '_idc'] = self.df[col].apply(lambda x: idc_reverse(x))

        if self.drop_origin:
            self.drop(col)

    # test passed
    def cut(self, col, bins, labels=False):
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
        self.df[col + '_cut'], bins_df1 = pd.cut(self.df[col], bins=bins, labels=labels, retbins=True)
        if self.drop_origin:
            self.drop(col)

    # test passed
    def qcut(self, col, q, labels=False):
        """
        Cut a column based on quantiles
        :param col: column name
        :param q: number of bins, each bin gets same nb of samples
        :param labels: list of label, len(labels) == q
        :param drop: drop the origin column
        """
        self.df[col + '_qcut'], bins_df1 = pd.qcut(self.df[col], q=q, labels=labels, retbins=True)
        if self.drop_origin:
            self.drop(col)

    def general_encoder(self, num_cols=None, ordinal_cols=None, one_hot_cols=None, drop=None):
        """
        A general encoder to transform numerical, categorical and ordinal features.
        :param drop:
                - None : retain all features (the default).
                - 'first' : drop the first category in each feature. If only one
                            category is present, the feature will be dropped entirely.
                - 'if_binary' : drop the first category in each feature with two
                                categories. Features with 1 or more than 2 categories are
                                left intact.
        :param num_cols: list of numerical columns to be encoded
        :param one_hot_cols: list of categorical columns to be one-hot encoded
        :param ordinal_cols: list of ordinal columns to be ordinal encoded
        :return
            encoded df
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
            cls_cat1 = ohe.fit_transform(self.df[one_hot_cols]).toarray()
            df1_cat = pd.DataFrame(data=cls_cat1, columns=ohe.get_feature_names(one_hot_cols).tolist())

        if num_cols is None:
            num_cols = []
            df1_num = pd.DataFrame()
        else:
            cls_num1 = ss.fit_transform(self.df[num_cols])
            df1_num = pd.DataFrame(cls_num1, columns=num_cols)

        if ordinal_cols is None:
            ordinal_cols = []
            df1_ord = pd.DataFrame()
        else:
            cls_ord1 = oe.fit_transform(self.df[ordinal_cols])
            df1_ord = pd.DataFrame(cls_ord1, columns=ordinal_cols)

        cls_rest = list(set(self.df.columns.tolist()) - set(num_cols) - set(one_hot_cols) - set(ordinal_cols))

        df_encoded = pd.concat(
            [df1_num.reset_index(drop=True), df1_cat.reset_index(drop=True), df1_ord.reset_index(drop=True),
             self.df[cls_rest].copy().reset_index(drop=True)], axis=1)

        return df_encoded


class FeatureHandling2(FeatureHandling):
    """
    Several feature handling methods.
    1st dataframe (usually training set) will be fit then transformed.
    2st dataframe (usually test set) will be transformed based on the 1st dataframe.
    """

    def __init__(self, df, df2, drop_origin=True):
        """
        :param df: a dataframe to be fit then transformed, usually the training set
        :param df: a dataframe to be transformed, usually the test set
        :param drop_origin: drop origin feature if new ones are created
        """
        super().__init__(df, drop_origin)
        self.df2 = df2.copy()

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
                temp = self.df[col].mean()
            elif method == 'median':
                temp = self.df[col].median()
            elif method == 'mode':
                temp = self.df[col].mode()[0]
            else:
                temp = method
            self.df[col].fillna(temp, inplace=True)
            self.df2[col].fillna(temp, inplace=True)

        else:
            if method == 'mean':
                self.df[col] = self.df.groupby(groupby)[col].apply(lambda x: x.fillna(x.mean()))
            elif method == 'median':
                self.df[col] = self.df.groupby(groupby)[col].apply(lambda x: x.fillna(x.median()))
            elif method == 'mode':
                self.df[col] = self.df.groupby(groupby)[col].apply(lambda x: x.fillna(x.mode()[0]))
            else:
                print("Please set 'groupby' to None!")
            # todo: transform on df2

    def manual_imputers(self, method_dict):
        """
        Impute missing values in several columns at a time, must specify the imputing method for each column
        :param method_dict: a dictionary indicating impute method for each column
                            eg: {col1:'mode', col2:'mean', col3:'other'}
        """
        for col in method_dict.keys():
            self.imputer(col, method_dict[col])

    def auto_imputers(self, cat_method='mode', num_method='median'):
        """
        Impute missing values in several columns at a time, automatically detect categorical and numerical features
        :param cat_method: imputing method for categorical features
        :param num_method: imputing method for numerical features
        """
        for col in self.df.select_dtypes('object').columns.tolist():
            self.imputer(col, cat_method)
        for col in self.df.select_dtypes('number').columns.tolist():
            self.imputer(col, num_method)

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

        top_n = super().keep_top_n(col, N, include_nan, result)
        self.df2[col] = self.df2[col].apply(lambda x: x if x in top_n else result)

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
        upper_bound, lower_bound = super().handle_outlier(col)
        if not drop_row:
            self.df2.loc[(self.df[col] < lower_bound) | (self.df[col] > upper_bound), col] = np.nan
        else:
            self.df2 = self.df2[(self.df2[col] <= upper_bound) & (self.df2[col] >= lower_bound)]

    def outlier_encoder(self, col, standard_scaling=True, method='cut'):
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

        q1 = self.df[col].quantile(q=0.25)
        q3 = self.df[col].quantile(q=0.75)
        iqr = q3 - q1
        upper_bound = 1.5 * iqr + q3
        lower_bound = q1 - 1.5 * iqr
        std = self.df[col].std()
        mean = self.df[col].mean()
        median = self.df[col].median()

        def extract_lower_bound(x):
            return 1 if x < lower_bound else 0

        def extract_upper_bound(x):
            return 1 if x > upper_bound else 0

        def cut_outliers(x):
            if x > upper_bound:
                return upper_bound
            elif x < lower_bound:
                return lower_bound
            return x

        def mean_outliers(x):
            return mean if x > upper_bound or x < lower_bound else x

        def median_outliers(x):
            return median if x > upper_bound or x < lower_bound else x

        def mean_std_scaling(x):
            return (x - mean) / std

        self.df[col + '_lower'] = self.df[col].apply(lambda x: extract_lower_bound(x))
        self.df[col + '_upper'] = self.df[col].apply(lambda x: extract_upper_bound(x))
        if method == 'cut':
            self.df[col + '_inner'] = self.df[col].apply(lambda x: cut_outliers(x))
        elif method == 'mean':
            self.df[col + '_inner'] = self.df[col].apply(lambda x: mean_outliers(x))
        elif method == 'median':
            self.df[col + '_inner'] = self.df[col].apply(lambda x: median_outliers(x))
        if standard_scaling:
            self.df[col + '_inner'] = self.df[col].apply(lambda x: mean_std_scaling(x))

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

        if self.drop_origin:
            self.drop(col)

    def handle_skewness(self, col):
        """
        log(1+x) transformation if col>0 and -1<skewness<1
        Only do this for one DataFrame: df1
        :param col: column name, must be numerical
        """
        skewness = self.get_skewness(self.df, col)
        print('Skewness: ' + str(skewness))

        if skewness <= 1 or skewness >= -1:
            print('No need of transformation')
        else:
            min_value = self.df[col].min()
            if min_value < 0:
                # move within x-axis
                self.df[col + '_log1x'] = self.df[col] + min_value
            # log(1+x) transformation
            self.df[col + '_log1x'] = np.log1p(self.df[col])

            if self.df2.min() < min_value:
                print("Failed to handle skewness in df2")
                return
            elif min_value < 0:
                self.df2[col + '_log1x'] = self.df2[col] + min_value
            self.df2[col + '_log1x'] = np.log1p(self.df2[col])

            print('Skewness: ' + self.get_skewness(self.df, col + '_log1x'))

            if self.drop_origin:
                self.drop(col)

    # test passed
    def binning(self, col, threshold, reverse=False):
        """
        Binarize a column based on a threshold
        :param col: column name
        :param threshold: a threshold to split data
        :param reverse:
                False: output = 1 when value is larger than threshold
                True: output = 1 when value is smaller than threshold
        """

        def bin_(x):
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
            self.df[col + '_binned'] = self.df[col].apply(lambda x: bin_(x))
            self.df2[col + '_binned'] = self.df2[col].apply(lambda x: bin_(x))
        else:
            self.df[col + '_binned'] = self.df[col].apply(lambda x: bin_reverse(x))
            self.df2[col + '_binned'] = self.df2[col].apply(lambda x: bin_reverse(x))

        if self.drop_origin:
            self.drop(col)

    # test passed
    def drop(self, col):
        super().drop(col)
        self.df2.drop(col, axis=1, inplace=True)

    # test passed
    def indicator(self, col, target=np.nan, reverse=False):
        """
        Create an indicator column to indicate the presence of a specific value
        :param col: column name
        :param target: the specific value for creating indicator
        :param reverse:
                False: value == target, then gives 1
                True: value == target, then gives 0
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
            self.df[col + '_idc'] = self.df[col].apply(lambda x: idc(x))
            self.df2[col + '_idc'] = self.df2[col].apply(lambda x: idc(x))
        else:
            self.df[col + '_idc'] = self.df[col].apply(lambda x: idc_reverse(x))
            self.df2[col + '_idc'] = self.df2[col].apply(lambda x: idc_reverse(x))

        if self.drop_origin:
            self.drop(col)

    # test passed
    def cut(self, col, bins, labels=False):
        """
        Cut a column based on number of bins or customized bin edges.
        Should not contain missing values, or they will be imputed by median.
        :param col: column name
        :param bins:
                number of bins: set number of bins, each bin has the same range size
                list of bin edges: set bin edges manually
        :param labels: list of label, len(labels) == bins or len(bin)-1
        """

        self.imputer(col, 'median')
        self.df[col + '_cut'], bins_df1 = pd.cut(self.df[col], bins=bins, labels=labels, retbins=True)
        self.df2[col + '_cut'] = pd.cut(self.df2[col], bins=bins_df1, labels=labels)
        if self.drop_origin:
            self.drop(col)

    # test passed
    def qcut(self, col, q, labels=False):
        """
        Cut a column based on quantiles
        :param col: column name
        :param q: number of bins, each bin gets same nb of samples
        :param labels: list of label, len(labels) == q
        """
        self.df[col + '_qcut'], bins_df1 = pd.qcut(self.df[col], q=q, labels=labels, retbins=True)
        self.df2[col + '_qcut'] = pd.cut(self.df2[col], bins=bins_df1, labels=labels)
        if self.drop_origin:
            self.drop(col)

    def general_encoder(self, num_cols=None, ordinal_cols=None, one_hot_cols=None, return_encoders=False, drop=None):
        """
        A general encoder to transform numerical, categorical and ordinal features.
        :param drop: for one-hot encoding
                - None : retain all features (the default).
                - 'first' : drop the first category in each feature. If only one
                            category is present, the feature will be dropped entirely.
                - 'if_binary' : drop the first category in each feature with two
                                categories. Features with 1 or more than 2 categories are
                                left intact.
        :param num_cols: list of numerical columns to be encoded
        :param one_hot_cols: list of categorical columns to be one-hot encoded
        :param ordinal_cols: list of ordinal columns to be ordinal encoded
        :param return_encoders: return these 3 encoders if True
        :return
            encoded df1 and df2
            encoded df1 and df2, StandardScaler, OrdinalEncoder, OneHotEncoder
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
            df2_cat = pd.DataFrame()
        else:
            cls_cat1 = ohe.fit_transform(self.df[one_hot_cols]).toarray()
            df1_cat = pd.DataFrame(data=cls_cat1, columns=ohe.get_feature_names(one_hot_cols).tolist())
            cls_cat2 = ohe.transform(self.df2[one_hot_cols]).toarray()
            df2_cat = pd.DataFrame(data=cls_cat2, columns=ohe.get_feature_names(one_hot_cols).tolist())
        if num_cols is None:
            num_cols = []
            df1_num = pd.DataFrame()
            df2_num = pd.DataFrame()
        else:
            cls_num1 = ss.fit_transform(self.df[num_cols])
            df1_num = pd.DataFrame(cls_num1, columns=num_cols)
            cls_num2 = ss.transform(self.df2[num_cols])
            df2_num = pd.DataFrame(cls_num2, columns=num_cols)

        if ordinal_cols is None:
            ordinal_cols = []
            df1_ord = pd.DataFrame()
            df2_ord = pd.DataFrame()
        else:
            cls_ord1 = oe.fit_transform(self.df[ordinal_cols])
            df1_ord = pd.DataFrame(cls_ord1, columns=ordinal_cols)
            cls_ord2 = oe.transform(self.df2[ordinal_cols])
            df2_ord = pd.DataFrame(cls_ord2, columns=ordinal_cols)

        cls_rest = list(set(self.df.columns.tolist()) - set(num_cols) - set(one_hot_cols) - set(ordinal_cols))

        df_encoded = pd.concat(
            [df1_num.reset_index(drop=True), df1_cat.reset_index(drop=True), df1_ord.reset_index(drop=True),
             self.df[cls_rest].copy().reset_index(drop=True)], axis=1)
        df2_encoded = pd.concat(
            [df2_num.reset_index(drop=True), df2_cat.reset_index(drop=True), df2_ord.reset_index(drop=True),
             self.df2[cls_rest].copy().reset_index(drop=True)], axis=1)

        if return_encoders:
            return df_encoded, df2_encoded, ss, oe, ohe
        else:
            return df_encoded, df2_encoded
