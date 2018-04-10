#!/usr/bin/env python

import os
import re
from bs4 import BeautifulSoup
try:
    from nltk.corpus import stopwords
except LookupError:
    import nltk.downloader
    nltk.downloader.download('stopwords')
    from nltk.corpus import stopwords


def review_to_wordlist(review, remove_stopwords=False, remove_punc=True, remove_num=True) -> list:
    """
    Function to convert a document to a sequence of words, optionally removing stop words.
    Returns a list of words.
    """

    review_text = BeautifulSoup(review, 'html.parser').get_text()

    if remove_punc:
        review_text = re.sub("[^a-zA-Z0-9]", " ", review_text)

    review_text = re.sub("[0-9]+", lambda _: " " if remove_num else "NUM", review_text)

    words = review_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return words


def review_to_sentences(review, tokenizer, remove_stopwords=False) -> list:
    """
    Function to split a review into parsed sentences. Returns a list of sentences, where each sentence is a list of
    words
    """
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.decode('utf8').strip())

    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences
