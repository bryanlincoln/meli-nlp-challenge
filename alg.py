from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split


def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer


def run(x_train_total, y_train_total, x_test_total, params):
    print('Splitting total dataset...')
    x_train, x_test, y_train, y_test = train_test_split(
        x_train_total, y_train_total, test_size=0.2, random_state=40)

    print('Generating TFIDF')
    x_train_tfidf, tfidf_vectorizer = tfidf(x_train)
    x_test_tfidf = tfidf_vectorizer.transform(x_test)

    print('Fitting LogisticRegression')
    clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                                   multi_class='multinomial', n_jobs=-1, random_state=40)
    clf_tfidf.fit(x_train_tfidf, y_train)

    print('Evaluating...')
    y_predicted_tfidf = clf_tfidf.predict(x_test_tfidf)

    accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(
        y_test, y_predicted_tfidf)

    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" %
          (accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf))
