import argparse
import sys
import torch
import embeddings
import preprocess
from utils import seed_everything
from alg import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Category Classifier')
    parser.add_argument('--embed_size', type=int,
                        default=300, help='How big is each word vector')
    parser.add_argument('--max_features', type=int,
                        default=120000, help='Num rows in embedding vector')
    parser.add_argument('--maxlen', type=int,
                        default=70, help='Max length of a sequence')
    parser.add_argument('--batch_size', type=int,
                        default=512, help='How many samples to process at once')
    parser.add_argument('--n_epochs', type=int,
                        default=10, help='How many times to iterate over all samples')
    parser.add_argument('--n_splits', type=int,
                        default=2, help='Number of K-fold Splits')
    parser.add_argument('--SEED', type=int,
                        default=10, help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Should the whole data be loaded?')
    parser.add_argument('--lang', type=str,
                        default='pt', help='pt or es')
    parser.add_argument('--loss_fn',
                        default=torch.nn.BCEWithLogitsLoss(reduction='sum'),
                        help='How big is each word vector')
    parser.add_argument('--preprocess', action='store_true',
                        help='Redo preprocessing.')
    parser.add_argument('--embed', action='store_true',
                        help='Redo embedding preprocessing.')

    # Setup
    params = parser.parse_args()

    if params.debug:
        print('Running in debug mode.')

    seed_everything()

    if params.preprocess:
        x_train, x_test, y_train, features, test_features, word_index = preprocess.preprocess(
            params)
    else:
        x_train, x_test, y_train, features, test_features, word_index = preprocess.load(
            params)

    if params.embed:
        embedding_matrix = embeddings.process(params)
    else:
        embedding_matrix = embeddings.load(params)

    preds = run(x_train, y_train, features, test_features,
                x_test, embedding_matrix, params)
