import numpy as np
import pickle
from utils import seed_everything


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = []
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        try:
            embedding = np.array([float(val) for val in splitLine[-300:]])
            model.append(embedding.reshape(300,))
        except:
            print(line[:20])
            continue

    print("Done.", np.shape(model)[0], " words loaded!")
    return model


def process(params):
    print('Loading embeddings from glove file.')

    seed_everything()

    if params.debug:
        embedding_matrix = np.random.randn(120000, 300)
    else:
        embedding_matrix = loadGloveModel(
            '../embeddings/glove_s300_' + params.lang + '.txt')

        with open("embedding_" + params.lang + ".pkl", "wb") as f:
            pickle.dump(embedding_matrix, f)
            f.close()

    print('Done loading embeddings.')

    return embedding_matrix


def load(params):
    if params.debug:
        return process(params)

    try:
        file = open("embedding_" + params.lang + ".pkl", "rb")
        print('Loading embeddings from pickle')
        return pickle.load(file)
    except FileNotFoundError:
        return process(params)
