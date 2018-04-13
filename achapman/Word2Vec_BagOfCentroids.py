#!/usr/bin/env python

#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 2 of the tutorial and covers Bag of Centroids
#  for a Word2Vec model. This code assumes that you have already
#  run Word2Vec and saved a model called "300features_40minwords_10context"
#
# *************************************** #

try:
    from gensim.models import Word2Vec
except UserWarning:
    pass

from sklearn.cluster import KMeans
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import KaggleWord2VecUtility as kutil

data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def create_bag_of_centroids(wordlist, word_centroid_map):
    # The number of clusters is equal to the highest cluster index in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    # Pre-allocate the vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    # Loop over the words in the review. If the word is in the vocabulary, find which cluster it belongs to
    #  and increment that cluster count by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids


if __name__ == '__main__':
    model = Word2Vec.load("300features_40minwords_10context")

    # ****** Run k-means on the word vectors and print a few clusters
    # Set "k" (num_clusters) to be 1/5th of the vocabulary size for an average of 5 words per cluster
    word_vectors = model.wv.syn0
    num_clusters = word_vectors.shape[0] // 5

    print("Running K means")
    start = time.time()
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)
    print("Time taken for K Means clustering: ", time.time() - start, "seconds")

    # Create a Word / Index dictionary, mapping each vocabulary word to a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))

    # Print the first ten clusters
    for cluster in range(10):
        print("\nCluster %d" % cluster)
        words = []
        for i in range(len(word_centroid_map.values())):
            if word_centroid_map.values()[i] == cluster:
                words.append(word_centroid_map.keys()[i])
        print(words)

    # Create clean_train_reviews and clean_test_reviews as we did before
    train = pd.read_csv(
        os.path.join(data_dir, 'labeledTrainData.tsv'),
        header=0,
        delimiter="\t",
        quoting=3,
    )
    test = pd.read_csv(
        os.path.join(data_dir, 'testData.tsv'),
        header=0,
        delimiter="\t",
        quoting=3,
    )

    print("Cleaning training reviews")
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(kutil.review_to_wordlist(review, remove_stopwords=True))

    print("Cleaning test reviews")
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(kutil.review_to_wordlist(review, remove_stopwords=True))

    # ****** Create bags of centroids
    train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")
    for i, review in enumerate(clean_train_reviews):
        train_centroids[i] = create_bag_of_centroids(review, word_centroid_map)

    test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")
    for i, review in enumerate(clean_test_reviews):
        test_centroids[i] = create_bag_of_centroids(review, word_centroid_map)

    # ****** Fit a random forest and extract predictions
    forest = RandomForestClassifier(n_estimators=100)

    # Fitting the forest may take a few minutes
    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(train_centroids, train["sentiment"])
    result = forest.predict(test_centroids)

    # Write the test results
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv(os.path.join(data_dir, "BagOfCentroids.csv"), index=False, quoting=3)
    print("Wrote BagOfCentroids.csv")
