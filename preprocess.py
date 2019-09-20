import pickle
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
    text2 = ''

    if params.lang == 'pt':
        STOPWORDS = nltk.corpus.stopwords.words("portuguese")
    else:
        STOPWORDS = nltk.corpus.stopwords.words("spanish")

    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 2:
            text2 += token + ' '

    text2 = text2[:-1]  # remove last space
    return text2


def parallelize_apply(df, func, colname, num_process, newcolnames):
    # takes as input a df and a function for one of the columns in df
    pool = Pool(processes=num_process)
    arraydata = pool.map(func, tqdm(df[colname].values))
    pool.close()

    newdf = pd.DataFrame(arraydata, columns=newcolnames)
    df = pd.concat([df, newdf], axis=1)
    return df


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, 4)
    pool = Pool(4)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# some fetures
def add_features(df):
    df['title'] = df['title'].apply(lambda x: str(x))
    df["lower_title"] = df["title"].apply(lambda x: x.lower())
    df['total_length'] = df['title'].apply(len)
    df['num_words'] = df.title.str.count('\S+')
    df['num_unique_words'] = df['title'].apply(
        lambda title: len(set(w for w in title.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
    return df


def preprocess(params):
    print('Start preprocessing...')
    if params.debug:
        train_df = pd.read_csv("../train.csv")[:800]
        test_df = pd.read_csv("../test.csv")[:200]
    else:
        train_df = pd.read_csv("../train.csv")
        test_df = pd.read_csv("../test.csv")

    train_df = train_df[train_df['language'] == (
        'portuguese' if params.lang == 'pt' else 'spanish')]
    test_df = test_df[test_df['language'] == (
        'portuguese' if params.lang == 'pt' else 'spanish')]

    print("Train shape : ", train_df.shape)
    print("Test shape : ", test_df.shape)

    train = parallelize_dataframe(train_df, add_features)
    test = parallelize_dataframe(test_df, add_features)

    # Clean the text
    train_df["title"] = train_df["title"].apply(
        lambda x: clean_text(params, x))
    test_df["title"] = test_df["title"].apply(lambda x: clean_text(params, x))

    # fill up the missing values
    train_X = train_df["title"].fillna("").values
    test_X = test_df["title"].fillna("").values

    features = train[['num_unique_words', 'words_vs_unique']].fillna(0)
    test_features = test[['num_unique_words', 'words_vs_unique']].fillna(0)

    # doing PCA to reduce network run times
    ss = StandardScaler()
    pc = PCA(n_components=5)
    ss.fit(np.vstack((features, test_features)))
    features = ss.transform(features)
    test_features = ss.transform(test_features)

    # Tokenize the sentences
    tokenizer = Tokenizer(num_words=params.max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    # Pad the sentences
    train_X = pad_sequences(train_X, maxlen=params.maxlen)
    test_X = pad_sequences(test_X, maxlen=params.maxlen)

    # Get the target values
    train_y = train_df['category'].values

    # shuffling the data
    np.random.seed(params.SEED)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    features = features[trn_idx]

    # save preprocessed data
    with open("preprocessed_objs_" + params.lang + ('_debug' if params.debug else '') + ".pkl", "wb") as f:
        pickle.dump((train_X, test_X, train_y, features,
                     test_features, tokenizer.word_index), f)
        f.close()

    print('Preprocessed data and saved pickle.')
    return train_X, test_X, train_y, features, test_features, tokenizer.word_index


def load(params):
    try:
        file = open("preprocessed_objs_" + params.lang +
                    ('_debug' if params.debug else '') + ".pkl", "rb")
        print('Loading preprocessed data.')
        return pickle.load(file)
    except FileNotFoundError:
        return preprocess(params)
