import pandas as pd
from feature_generation import FeatureGeneration as FG


def main():
    df = pd.read_csv('titanic.csv')
    df = df[['Pclass', 'Sex', 'Age']]
    print(df)
    df_new, new_cols = FG.explicit_pairwise_interactions(df)
    print(df_new)


if __name__ == "__main__":
    main()
