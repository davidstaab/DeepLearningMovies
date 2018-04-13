#!/usr/bin/env python

import re
from bs4 import BeautifulSoup

from nltk.corpus import stopwords
try:
    stopwords.words('english')
except LookupError:
    import nltk.downloader
    nltk.downloader.download('stopwords')

from nltk.tokenize import sent_tokenize
try:
    sent_tokenize(['Hello world'], 'english')
except LookupError:
    import nltk.downloader
    nltk.downloader.download('punkt')


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


def review_to_sentences(review, *args) -> list:
    """
    Function to split a review into parsed sentences. Returns a list of sentences, where each sentence is a list of
    words. *args is the same as for `review_to_wordlist()`
    """

    raw_sentences = sent_tokenize(review.decode('utf8').strip(), 'english')

    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence):
            sentences.append(review_to_wordlist(raw_sentence, *args))
    return sentences
