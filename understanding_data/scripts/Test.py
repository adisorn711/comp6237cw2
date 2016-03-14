import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.porter import *

tokens = [word for sent in nltk.sent_tokenize("Hello World, It's inspired by me. !!!!") for word in nltk.word_tokenize(sent)]

print tokens

filtered_tokens = []
# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
for token in tokens:
    if re.search('[a-zA-Z]', token):
        filtered_tokens.append(token)

print filtered_tokens

stemmer = PorterStemmer()
stems = [stemmer.stem(t) for t in filtered_tokens]

print stems


