from dataframe_observer import *
from feature_handling import *
from feature_reduction import *
import os

DF_PATH = '../data/california_train.csv'
OUTPUT_PATH = '../report_california_SalePrice/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

TARGET = 'SalePrice'
TOP_K_CORR_NUM_PAIRS = 5
CORR_NUM_PLOT_SIZE = (31, 30)
TOP_K_MISSING_PATTERNS = 10
MISSING_PATTERN_PLOT_SIZE = (8, 10)
TOP_K_FEATURES = 10
UNIQUE_PERCENTAGE_THRESHOLD = 0.01  # set to None to skip auto type switch

# df summary
df = pd.read_csv(DF_PATH)
df_info = read_df_info(df, return_result=True)

# identify col types
target_col = TARGET
target_col_is_num = df[target_col].dtype != 'object'
id_cols = df_info[df_info['unique%'] == 1].index.tolist()

# auto type switch
if UNIQUE_PERCENTAGE_THRESHOLD is not None:
    high_unique_cols = df_info[df_info['unique%'] < UNIQUE_PERCENTAGE_THRESHOLD].index.tolist()
    for col in high_unique_cols:
        if df[col].dtype != 'object':
            print(col + ' is set to numerical')
            df[col] = df[col].astype('object')

cat_cols = df.select_dtypes('object').columns.tolist()
num_cols = df.select_dtypes('number').columns.tolist()
cat_cols = set(cat_cols) - set(id_cols) - set(target_col)
num_cols = set(num_cols) - set(id_cols) - set(target_col)
print(str(len(cat_cols)) + ' cat cols:')
print(cat_cols)
print(str(len(num_cols)) + ' num cols:')
print(num_cols)

writer = pd.ExcelWriter(OUTPUT_PATH + "report.xlsx")

df_info = read_df_info(df, return_result=True)
df_info.to_excel(writer, sheet_name="df_info")

# cat col stats
cat_col_stats = []
for col in cat_cols:
    stats, _ = describe_cat_col(df, col, return_result=True)
    cat_col_stats.append(stats)
cat_col_stats = pd.DataFrame(cat_col_stats)
cat_col_stats.set_index('Name', inplace=True)
cat_col_stats.to_excel(writer, sheet_name="cat_cols_stats")

# num col stats
num_col_stats = []
for col in num_cols:
    stats = describe_num_col(df, col, return_result=True)
    num_col_stats.append(stats)
num_col_stats = pd.DataFrame(num_col_stats)
num_col_stats.set_index('Name', inplace=True)
num_col_stats.to_excel(writer, sheet_name="num_cols_stats")

# corr for num cols
pairs, corr = correlation(
    df[list(num_cols)], k=TOP_K_CORR_NUM_PAIRS, plot_size=CORR_NUM_PLOT_SIZE, file_name=OUTPUT_PATH + 'CORR_NUM_NUM.png')
corr.to_excel(writer, sheet_name="num_cols_corr")

num_num_results = []
for col1, col2, _, _ in pairs:
    result = describe_num_num(
        df, col1, col2, return_result=True, file_name=OUTPUT_PATH + 'HIGHCORR_' + col1 + '_' + col2 + '.png')
    num_num_results.append(result)
num_num_results = pd.DataFrame(num_num_results)
num_num_results.to_excel(writer, sheet_name="high_corr_num_num")

# missing patterns
missing_pattern_df = missing_pattern(
    df,
    TOP_K_MISSING_PATTERNS,
    plot_size=MISSING_PATTERN_PLOT_SIZE,
    file_name=OUTPUT_PATH + 'MISSING_PATTERN.png'
)
missing_pattern_df.to_excel(writer, sheet_name="missing_pattern")

# target
target_stat = auto_describe_col(
    df, target_col,
    file_name=OUTPUT_PATH + 'TARGET_' + target_col + '.png', return_result=True
)
target_series = pd.Series(target_stat)
target_series.to_excel(writer, sheet_name="target")

# best features (by RF)
X = df[list(num_cols) + list(cat_cols)].copy()
X = auto_imputers(X)
print('X.columns')
print(X.columns.tolist())
X = general_encoder(X, num_cols=num_cols, ordinal_cols=cat_cols, one_hot_cols=None)
print('X.columns')
print(X.columns.tolist())
X = tree_based(X, df[target_col], regression=target_col_is_num, n_keep=TOP_K_FEATURES)
important_features = X.columns.tolist()

print('important_features')
print(important_features)

imt_feat_results = []
for feature in important_features:
    print('feature:')
    print(feature)
    print(df[feature].value_counts())
    auto_describe_col(df, feature, file_name=OUTPUT_PATH + 'IMP_FEAT_' + feature + '.png', return_result=True)
    result = auto_describe_pair(
        df, col1=target_col, col2=feature,
        return_result=True,
        file_name=OUTPUT_PATH + 'TGTFEAT_' + feature + '.png'
    )
    imt_feat_results.append(result)
imt_feat_results = pd.DataFrame(imt_feat_results)
imt_feat_results.to_excel(writer, sheet_name="target_imp_feats")


writer.save()
