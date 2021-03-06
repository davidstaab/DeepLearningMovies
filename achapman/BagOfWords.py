#!/usr/bin/env python

#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 1 of the tutorial on Natural Language Processing.
#
# *************************************** #
#  Updates and changes: David Staab
#  Date: April 2018

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import KaggleWord2VecUtility as kutil
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    train = pd.read_csv(os.path.join(data_dir, 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)
    test = pd.read_csv(os.path.join(data_dir, 'testData.tsv'), header=0, delimiter="\t", quoting=3)

    print('The first review is:')
    print(train["review"][0])

    input("Press Enter to continue...")

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length of the movie review list
    print("Cleaning and parsing the training set movie reviews...\n")
    for i in range(len(train["review"])):
        clean_train_reviews.append(" ".join(kutil.review_to_wordlist(train["review"][i], remove_stopwords=True)))

    print("Creating the bag of words...\n")
    # the "CountVectorizer" object is scikit-learn's bag of words tool.
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    np.asarray(train_data_features)

    # ******* Train a random forest using the bag of words
    print("Training the random forest (this may take a while)...")
    forest = RandomForestClassifier(n_estimators=100)

    # Fit the forest to the training set, using the bag of words as
    # features and the sentiment labels as the response variable
    forest = forest.fit(train_data_features, train["sentiment"])

    print("Cleaning and parsing the test set movie reviews...\n")
    clean_test_reviews = []
    for i in range(len(test["review"])):
        clean_test_reviews.append(" ".join(kutil.review_to_wordlist(test["review"][i], remove_stopwords=True)))

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    np.asarray(test_data_features)

    # Use the random forest to make sentiment label predictions
    print("Predicting test labels...\n")
    result = forest.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

    # Use pandas to write the comma-separated output file
    output.to_csv(os.path.join(data_dir, 'Bag_of_Words_model.csv'), index=False, quoting=3)
    print("Wrote results to Bag_of_Words_model.csv")
