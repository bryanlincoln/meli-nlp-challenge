import pickle
import random
import re
from multiprocessing import Pool
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from gensim.utils import simple_preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download("stopwords")


def clean_text(params, text):
    text = str(text)
    text2 = ""

    if params.lang == "pt":
        STOPWORDS = nltk.corpus.stopwords.words("portuguese")
    else:
        STOPWORDS = nltk.corpus.stopwords.words("spanish")

    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 2:
            text2 += token + " "

    text2 = text2[:-1]  # remove last space
    return text2


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, 4)
    pool = Pool(4)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# some fetures
def add_features(df):
    df["title"] = df["title"].apply(lambda x: str(x))
    df["lower_title"] = df["title"].apply(lambda x: x.lower())
    df["total_length"] = df["title"].apply(len)
    df["num_words"] = df.title.str.count("\S+")
    df["num_unique_words"] = df["title"].apply(
        lambda title: len(set(w for w in title.split())))
    df["words_vs_unique"] = df["num_unique_words"] / df["num_words"]
    return df


def preprocess(params):
    print("Start preprocessing...")
    if params.debug:
        train_df = pd.read_csv("../train.csv")[:800]
        test_df = pd.read_csv("../test.csv")[:200]
    elif params.subset:
        test_df = pd.read_csv("../test.csv")

        filename = "../train.csv"
        # number of records in file (excludes header)
        n = sum(1 for line in open(filename)) - 1
        # the 0-indexed header will not be included in the skip list
        skip = sorted(random.sample(range(1, n + 1), n - params.subset_size))
        train_df = pd.read_csv(filename, skiprows=skip)
    else:
        train_df = pd.read_csv("../train.csv")
        test_df = pd.read_csv("../test.csv")

    """train_df = train_df[train_df["language"] == (
        "portuguese" if params.lang == "pt" else "spanish")]
    test_df = test_df[test_df["language"] == (
        "portuguese" if params.lang == "pt" else "spanish")]"""

    # train_df = train_df[train_df["label_quality"] == "reliable"]
    print(train_df.head())
    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

    # Clean the text
    train_df["title"] = train_df["title"].apply(
        lambda x: clean_text(params, x))
    test_df["title"] = test_df["title"].apply(lambda x: clean_text(params, x))

    print(train_df.head())
    print(test_df.head())

    # save preprocessed data
    train_df.to_csv("../processed_train_" + params.lang +
                    ("_debug" if params.debug else "") + ".csv", index=False)
    test_df.to_csv("../processed_test_" + params.lang +
                   ("_debug" if params.debug else "") + ".csv", index=False)

    print("Preprocessed data and saved to csv.")
    return train_df["title"], train_df["category"], test_df["title"]


def load(params):
    print("Start loading...")
    if params.debug:
        train_df = pd.read_csv("../processed_train_" + params.lang +
                               ("_debug" if params.debug else "") + ".csv")[:800]
        test_df = pd.read_csv("../processed_test_" + params.lang +
                              ("_debug" if params.debug else "") + ".csv")[:200]
    else:
        train_df = pd.read_csv(
            "../processed_train_" + params.lang + ("_debug" if params.debug else "") + ".csv")
        test_df = pd.read_csv(
            "../processed_test_" + params.lang + ("_debug" if params.debug else "") + ".csv")

    return train_df["title"], train_df["category"], test_df["title"]
