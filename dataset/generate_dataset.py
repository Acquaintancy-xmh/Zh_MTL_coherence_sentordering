import pandas as pd
from sklearn.utils import shuffle

def gen_test_dataset(path):
    df = pd.read_csv(path)
    df = shuffle(df)
    df_train = df.head(7094)
    df_valid = df.head(2000)
    df_test = df.head(2000)
    df_train.to_csv("train_fold_0.csv", index=False)
    df_valid.to_csv("valid_fold_0.csv", index=False)
    df_test.to_csv("test_fold_0.csv", index=False)

gen_test_dataset("labeled_corpus.csv")