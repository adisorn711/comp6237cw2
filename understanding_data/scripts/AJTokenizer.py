import os
import sys
from bs4 import BeautifulSoup
import pandas as pd
import re
import json
import string
import collections
import pprint
import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from scipy.cluster.hierarchy import ward, dendrogram

import matplotlib.pyplot as plt
import matplotlib as mpl

MDS()
en_words = set(words.words())
en_stopwords = set(stopwords.words("english"))

titles = ['Dictionary of Greek and Roman geography',
          'The History of Tacitus Book I',
          'The history of the Peloponnesia war Vol. II',
          'The history of Rome Vol. I',
          'The history of the decline and fall of the Roman Empire Vol. III',
          'The whole genuine works of Flavius Josephus Vol. II',
          'The Description of Greece',
          'LIVY Vol. III',
          'Gibbon\'s History of the Decline and Fall of The Roman Empire Vol. IV',
          'Gibbon\'s History of the Decline and Fall of The Roman Empire Vol. II',
          'The Historical Annals of Cornelius Tacitus Vol. I',
          'LIVY Vol. V',
          'The whole genuine works of Flavius Josephus Vol. I',
          'The history of the decline and fall Vol. V',
          'The Histories of Caius Cornelius Tacitus',
          'The Decline and Fall of the Roman Empire',
          'The history of the decline and fall of the Roman Empire Vol. I',
          'The Revolution Book IV',
          'The history of the Peloponnesia war Vol. I',
          'Titus Livius\' Roman History',
          'The works of Josephus Vol. IV',
          'The works of Cornelius Tacitus Vol. IV',
          'The first and Thirty-Third books of Pliny\'s Natural History',
          'The works of Josephus Vol. III']

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

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

class AJBaseTextObject(object):

    def __init__(self, title, contents, raw_text=''):
        self.__title = title
        self.__contents = contents
        self.__sortedWords = None
        self.__raw_texts = raw_text

    @property
    def sortedWords(self):
        if self.__sortedWords is None:
            pass

        return self.__sortedWords

    @property
    def words(self):
        return self.__contents.keys()

    @property
    def texts(self):
        return self.__raw_texts.decode("utf-8")

class AJTextClassification(object):

    def __init__(self):
        self.__books = []
        self.__no_of_features = 0
        self.__bag_of_words = set()
        self.__indexed_words = dict()  # {'word' : int column}
        self.__total_vocabs = []
        self.__tf_idf_matrix = None
        self.__dist_matrix = None

    def add_book(self, book):
        self.__books.append(book)

    def compute_max_features(self):
        features = []
        for book in self.__books:
            features.append(len(book.words))
        if len(features) == 0:
            self.__no_of_features = 0
        else:
            self.__no_of_features = max(features)

    def print_word_count(self):
        _words = []
        for book in self.__books:
            text_arr = book.texts.strip().split()
            count = 0
            for w in text_arr:
                if len(w) > 0:
                    count += 1

            _words.append(count)

        print _words

    def print_word_count_after_tokenization(self):
        _words = []
        for book in self.__books:
            text = book.texts.decode("utf-8").strip()
            res = tokenize_and_stem(text)
            _words.append(len(res))

        print _words


    # read from json and return in Dictionary { Txt : Int }
    def read_data_from_path(self, data_path):
        print 'read data from {}'.format(data_path)
        with open(data_path) as data_file:
            data = json.load(data_file)
            #print len(data)
            data_file.close()
        return data

    def read_raw_text_from_path(self, data_path):
        print 'read data from {}'.format(data_path)
        with open(data_path) as data_file:
            data = data_file.read()
            #print len(data)
            data_file.close()
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
            #letters_only = re.sub("[^a-zA-Z]", " ", word)
            #lower_word = letters_only.lower().strip()

            if word not in stops and len(word) > 0:
                #res.append(lower_word)
                res[word] = freq

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

            if len(json_files) > 0:
                f = ''
                f_raw = ''
                for f_name in json_files:
                    if f_name.endswith('.json'):
                        f = f_name
                    if f_name.endswith('.txt'):
                        f_raw = f_name

                #dic = self.read_data_from_path(os.path.join(root_path, d, f))
                raw_txt = self.read_raw_text_from_path(os.path.join(root_path, d, f_raw))
                #print len(dic)
                #res = self.review_to_words(dic)
                #print len(raw_txt)

                book = AJBaseTextObject(f, {}, raw_text=raw_txt)
                self.add_book(book)

        self.compute_max_features()
        print "Max features is {}".format(self.__no_of_features)
        #self.print_word_count()
        #self.print_word_count_after_tokenization()

    def create_index_of_words(self):
        for book in self.__books:
            words = book.words
            print "#num = {}".format(len(words))
            for w in words:
                self.__bag_of_words.add(w)

        print "in bag: {}".format(len(self.__bag_of_words))
        sorted_words = sorted(list(self.__bag_of_words))
        print sorted_words[:10]

        self.__total_vocabs = sorted_words
        for index, item in enumerate(sorted_words):
            self.__indexed_words[item] = index

    def tf_idf(self):
        files = os.listdir('./')
        saved_file_name = 'doc_matrix.pkl'
        tfidf_matrix = None
        if saved_file_name not in files:
            #define vectorizer parameters
            tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=250000,
                                             min_df=0.01, stop_words='english',
                                             use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 1))
            """
            tfidf_vectorizer = CountVectorizer( analyzer = "word",   \
                                                tokenizer = tokenize_and_stem,    \
                                                preprocessor = None, \
                                                stop_words = 'english',   \
                                                lowercase=True, \
                                                max_features=250000)
            """

            books = []
            for book in self.__books:
                books.append(book.texts)

            tfidf_matrix = tfidf_vectorizer.fit_transform(books) #fit the vectorizer to synopses

            joblib.dump(tfidf_matrix, saved_file_name)
        else:
            tfidf_matrix = joblib.load(saved_file_name)

        self.tf_idf_matrix = tfidf_matrix
        print(self.tf_idf_matrix.shape)

        self.__dist_matrix = 1 - cosine_similarity(self.tf_idf_matrix)
        print self.__dist_matrix

        """
        synopes = []
        for book in self.__books:
            synopes.append(book.texts)
        #not super pythonic, no, not at all.
        #use extend so it's a big flat list of vocab
        for i in synopes:
            allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
            totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list

            allwords_tokenized = tokenize_only(i)
            totalvocab_tokenized.extend(allwords_tokenized)

        vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
        print 'there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame'
        """

    def k_mean_clustering(self, n_clusters=5):
        num_clusters = n_clusters

        km = KMeans(n_clusters=num_clusters)
        km.fit(self.tf_idf_matrix)
        clusters = km.labels_.tolist()
        print clusters


        synopses = []
        for book in self.__books:
            synopses.append(book.texts)

        books = { 'title': titles, 'synopsis': synopses, 'cluster': clusters}

        frame = pd.DataFrame(books, index = [clusters] , columns = ['title', 'cluster'])
        print frame['cluster'].value_counts()
        #grouped = frame['title'].groupby(frame['cluster']) #groupby cluster for aggregation purposes

        #print grouped.mean() #average rank (1 to 100) per cluster

        # convert two components as we're plotting points in a two-dimensional plane
        # "precomputed" because we provide a distance matrix
        # we will also specify `random_state` so the plot is reproducible.
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

        pos = mds.fit_transform(self.__dist_matrix)  # shape (n_components, n_samples)
        xs, ys = pos[:, 0], pos[:, 1]
        print xs
        print ys

        #set up colors per clusters using a dict
        cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

        #set up cluster names using a dict
        cluster_names = {0: 'Group 1',
                         1: 'Group 2',
                         2: 'Group 3',
                         3: 'Group 4',
                         4: 'Group 5'}

        #create data frame that has the result of the MDS plus the cluster numbers and titles
        df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

        #group by cluster
        groups = df.groupby('label')


        # set up plot
        fig, ax = plt.subplots(figsize=(17, 9)) # set size
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

        #iterate through groups to layer the plot
        #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                    label=cluster_names[name], color=cluster_colors[name],
                    mec='none')
            ax.set_aspect('auto')
            ax.tick_params(\
                axis= 'x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            ax.tick_params(\
                axis= 'y',         # changes apply to the y-axis
                which='both',      # both major and minor ticks are affected
                left='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelleft='off')

        ax.legend(numpoints=1)  #show legend with only 1 point

        #add label in x,y position with the label as the film title
        for i in range(len(df)):
            ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)

        plt.show() #show the plot

    def hierachical_clustering(self):
        linkage_matrix = ward(self.__dist_matrix) #define the linkage_matrix using ward clustering pre-computed distances

        fig, ax = plt.subplots(figsize=(15, 9)) # set size
        ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

        plt.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')

        fig.set_tight_layout(True) #show plot with tight layout
        plt.show()

        #uncomment the below to save the plot if need be
        #plt.savefig('clusters_small_noaxes.png', dpi=200)

    @property
    def tf_idf_matrix(self):
        return self.__tf_idf_matrix

    @tf_idf_matrix.setter
    def tf_idf_matrix(self, s):
        self.__tf_idf_matrix = s


stemmer = SnowballStemmer("english")

if __name__ == '__main__':
    my_classification = AJTextClassification()
    my_classification.run()
    my_classification.tf_idf()
    my_classification.k_mean_clustering(4)
    my_classification.hierachical_clustering()




