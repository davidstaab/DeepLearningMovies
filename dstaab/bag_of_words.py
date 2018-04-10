# My attempt at implementing Part 1
# Approach:
#   Use word multiplicities as features for training/estimation
#   Model estimator on a random forest classifier

import pandas as pd
import csv
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import numpy as np
import re

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def preprocess(raw: str) -> str:
    return BeautifulSoup(raw, 'html.parser').get_text()


def build_train_data(max_features: int=5000) -> tuple:
    """
    Extract feature vectors from source data.
    Return:
        samples (of those vectors)
        their labels
        vectorizer object used for feature extraction
    """
    train = pd.read_csv(os.path.join(data_dir, 'labeledTrainData.tsv'), delimiter='\t', quoting=csv.QUOTE_NONE)
    # Notes:
    # Default tokenizer uses 2+ alphanumeric chars as a token. Ignores all punctuation.
    vec = CountVectorizer(
        analyzer='word',
        preprocessor=preprocess,
        stop_words='english',
        max_features=max_features,
    )
    samples = vec.fit_transform(train['review'])
    labels = train['sentiment']
    return samples, labels, vec


def build_test_data(vectorizer) -> tuple:
    """
    Extract feature vectors from source data.
    Return:
         samples (of those vectors)
         document IDs sorted along rows of sample matrix
         original document texts (extracted from raw HTML) sorted same as IDs
    """
    test = pd.read_csv(os.path.join(data_dir, 'testData.tsv'), delimiter='\t', quoting=csv.QUOTE_NONE)
    # Notes:
    # See CountVectorizer notes from build_train_data()
    # Have to manually preprocess the data here
    docs = []
    for raw in test['review']:
        docs.append(preprocess(raw))
    return vectorizer.transform(docs), test['id'], docs


def build_model(n_estimators: int=100):
    return RandomForestClassifier(n_estimators=n_estimators)


def train_model(
        model: RandomForestClassifier,
        samples: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray=None,
):
    model.fit(samples, labels, sample_weight=sample_weights)


def test_model(model: RandomForestClassifier, samples: np.ndarray) -> np.ndarray:
    return model.predict(samples)


def report(data: dict):
    pd.DataFrame(data=data).to_csv(
        os.path.join(data_dir, 'Bag_of_Words_model.csv'),
        index=False,
        quoting=csv.QUOTE_NONE,
    )


if __name__ == '__main__':
    # TODO: Train on 80% of labeled data, validate on other 20%, generate visualization of prediction accuracy

    try:
        max_features = int(input('How many features do you want to train? (default 1000): '))
    except ValueError:
        print('Invalid input. Using default.')
        max_features = 1000
    x_train, y_train, vectorizer = build_train_data(max_features)

    try:
        n_estimators = int(input('How many estimators do you want to use in your random tree? (default 100): '))
    except ValueError:
        print('Invalid input. Using default.')
        n_estimators = 100
    model = build_model(n_estimators)

    train_model(model, x_train, y_train)
    print('Estimator parameters:\n' + re.sub(',\s', '\n', str(model.get_params(), ''))[1:-1])

    x_test, docs_id, docs_raw = build_test_data(vectorizer)
    predictions = test_model(model, x_test)

    report_data = {
        'id': docs_id,
        'sentiment': predictions,
    }
    report(report_data)
