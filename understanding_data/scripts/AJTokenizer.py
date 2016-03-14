import os
import sys
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
import pprint
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

class AJBaseTextObject(object):

    def __init__(self, title, contents):
        self.__title = title
        self.__contents = contents
        self.__sortedWords = None

    @property
    def sortedWords(self):
        if self.__sortedWords is None:
            pass

        return self.__sortedWords

    @property
    def words(self):
        return self.__contents.keys()

class AJTextClassification(object):

    def __init__(self):
        self.__books = []
        self.__no_of_features = 0

    def add_book(self, book):
        self.__books.append(book)

    def compute_max_features(self):
        features = []
        for book in self.__books:
            features.append(len(book.words))

        self.__no_of_features = max(features)

    # read from json and return in Dictionary { Txt : Int }
    def read_data_from_path(self, data_path):
        print 'read data from {}'.format(data_path)
        with open(data_path) as data_file:
            data = json.load(data_file)
            #print len(data)
        return data

    def review_to_words(self, raw_review_dic):
        # Function to convert a raw review to a string of words
        # The input is a single string (a raw movie review), and
        # the output is a single string (a preprocessed movie review)
        #
        # 1. Remove HTML
        #review_text = BeautifulSoup(raw_review).get_text()
        res = dict()
        stops = set(stopwords.words("english"))
        for word, freq in raw_review_dic.items():
            letters_only = re.sub("[^a-zA-Z]", " ", word)
            lower_word = letters_only.lower().strip()
            if lower_word not in stops:
                #res.append(lower_word)
                res[lower_word] = freq

        return res

    def run(self):
        root_path = '../db'
        dirs = os.listdir(root_path)
        directories = (x for x in dirs if os.path.isdir(os.path.join(root_path, x)))
        for d in directories:
            bag_dir = os.path.join(root_path, d)
            json_files = os.listdir(bag_dir)

            if ".DS_Store" in json_files:
                json_files.remove(".DS_Store")  # remove unwanted files in OS X system

            if len(json_files) == 1:
                f = json_files[0]
                dic = self.read_data_from_path(os.path.join(root_path, d, f))
                #print len(dic)
                res = self.review_to_words(dic)
                print len(res)

                book = AJBaseTextObject(f, res)
                self.add_book(book)

        self.compute_max_features()
        print "Max features is {}".format(self.__no_of_features)


stemmer = SnowballStemmer("english")

if __name__ == '__main__':
    my_classification = AJTextClassification()
    my_classification.run()






