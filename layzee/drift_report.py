import sys

sys.path.append("..")

from layzee.dataframe_observer import *
from layzee.feature_handling2 import *
from layzee.feature_drift import *
import os

DF1_PATH = '../data/california_train.csv'
DF2_PATH = '../data/california_test.csv'
OUTPUT_PATH = '../report_california_drift/'
TARGET = 'SalePrice'
UNIQUE_PERCENTAGE_THRESHOLD = 0.05  # set to None to skip auto type switch
ROC_TOLERANCE = 0.05
RANDOM_STATE = 1234

if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

writer = pd.ExcelWriter(OUTPUT_PATH + "report.xlsx")

# df summary
df1 = pd.read_csv(DF1_PATH)
df1_info = read_df_info(df1, return_result=True)

df2 = pd.read_csv(DF2_PATH)
df2_info = read_df_info(df2, return_result=True)

# drop target
target_col = TARGET
if target_col in df1.columns.tolist():
    df1.drop(target_col, axis=1, inplace=True)
if target_col in df2.columns.tolist():
    df2.drop(target_col, axis=1, inplace=True)

# drop id
id_cols = df1_info[df1_info['unique%'] == 1].index.tolist()
df1.drop(id_cols, axis=1, inplace=True)
df2.drop(id_cols, axis=1, inplace=True)

# auto type switch
if UNIQUE_PERCENTAGE_THRESHOLD is not None:
    high_unique_cols = df1_info[df1_info['unique%'] < UNIQUE_PERCENTAGE_THRESHOLD].index.tolist()
    for col in high_unique_cols:
        if df1[col].dtype != 'object':
            print(col + ' is set to numerical')
            df1[col] = df1[col].astype('object')
            df2[col] = df2[col].astype('object')

cat_cols = df1.select_dtypes('object').columns.tolist()
num_cols = df1.select_dtypes('number').columns.tolist()
cat_cols = set(cat_cols) - set(id_cols) - set(target_col)
num_cols = set(num_cols) - set(id_cols) - set(target_col)
print(str(len(cat_cols)) + ' cat cols:')
print(cat_cols)
print(str(len(num_cols)) + ' num cols:')
print(num_cols)

df1_info = read_df_info(df1, return_result=True)
df1_info.columns = ['dtype', 'unique_1', 'unique%_1', 'missing_1', 'missing%_1']
df2_info = read_df_info(df2, return_result=True)
df2_info.columns = ['dtype', 'unique_2', 'unique%_2', 'missing_2', 'missing%_2']
df2_info.drop('dtype', axis=1, inplace=True)

df_info = df1_info.join(df2_info)
df_info = df_info[[
    'dtype', 'unique_1', 'unique_2', 'unique%_1', 'unique%_2', 'missing_1', 'missing_2', 'missing%_1', 'missing%_2']]

df_info.to_excel(writer, sheet_name="df_info")

# stat detection
for col in num_cols.union(cat_cols):
    has_drift, result_df = auto_detection(
        df_train=df1, df_test=df2, col_name=col,
        file_name=OUTPUT_PATH + 'STAT_' + col + '.png', return_result=True)
    if has_drift:
        result_df = pd.DataFrame(result_df)
        result_df.to_excel(writer, sheet_name='STAT_' + col)

# adv detection
df1_ = df1.copy()
df2_ = df2.copy()

df1_, df2_ = auto_imputers(df1_, df2_)
df1_, df2_ = general_encoder(
    df1_, df2_,
    one_hot_cols=None,
    ordinal_cols=cat_cols,
    num_cols=num_cols
)

result = adversarial_detection(df1_, df2_, roc_tolerance=ROC_TOLERANCE, random_state=RANDOM_STATE)

for col in result:
    _, result_df = auto_detection(
        df_train=df1, df_test=df2, col_name=col,
        file_name=OUTPUT_PATH + 'ADV_' + col + '.png',
        return_result=True, adv_drift=True
    )
    result_df.to_excel(writer, sheet_name='ADV_' + col)

writer.save()
