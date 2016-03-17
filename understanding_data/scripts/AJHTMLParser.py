from HTMLParser import HTMLParser
import copy
import collections
import re
from nltk.corpus import words
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

en_words = set(words.words())
en_stopwords = set(stopwords.words("english"))

class CustomeCollection(object):

    def __init__(self):
        self.__dic = collections.OrderedDict()
        self.__raw_texts = []

    def add_object_for_key(self, key, obj):
        self.__dic[key] = obj

    def increase_value_for_key(self, key):
        if key in self.__dic:
            self.__dic[key] += 1
        else:
            self.__dic[key] = 1

    def clear_collection(self):
        self.__dic = dict()
        self.__raw_texts = []

    def append_raw_text(self, text):
        self.__raw_texts.append(text)

    @property
    def bag_of_words(self):
        return copy.deepcopy(self.__dic)

    @property
    def raw_texts(self):
        return self.__raw_texts

    @property
    def number_of_keys(self):
        return len(self.__dic.keys())


collection = CustomeCollection()
# create a subclass and override the handler methods
class SimpleParser(HTMLParser):

    #def __init__(self):
        #self.__bagOfWords = dict()

    def handle_starttag(self, tag, attrs):
        pass
        #print "Encountered a start tag:", tag

    def handle_endtag(self, tag):
        pass
        #print "Encountered an end tag :", tag

    def handle_data(self, data):
        d = data.strip()
        if len(d) >= 3:
            letters_only = re.sub("[^a-zA-Z]", " ", data)
            lower_word = letters_only.lower()

            # need to check inner spaces like 'a    xxxx'
            all_words = lower_word.split(' ')
            pre_processing_words = [w for w in all_words if len(w) >= 3 and w in en_words]
            #pre_processing_words = [w for w in all_words if len(w) >= 3]

            for w in pre_processing_words:
                if w not in en_stopwords:
                    collection.increase_value_for_key(w)
                    collection.append_raw_text(w)
            #print "Encountered some data  :", data


    def feed_parser(self, data):
        self.feed(data)

    def reset_parser(self):
        self.reset()

    def clear_bag(self):
        collection.clear_collection()

    @property
    def output(self):
        return collection.bag_of_words

    @property
    def raw_output(self):
        return collection.raw_texts

    @property
    def get_number_of_keys(self):
        return collection.number_of_keys

class BeautifulSoapParser(object):

    def __init__(self):
        self.__texts = []
        self.__html = ''

    def feed_parser(self, data):
        self.__html = data

        soup = BeautifulSoup(data, 'html.parser')
        self.__texts.append(soup.get_text().encode("utf-8"))

    def reset_parser(self):
        self.__html = ''

    def clear_bag(self):
        self.__texts = []
        self.__html = ''

    @property
    def output(self):
        return collection.bag_of_words

    @property
    def raw_output(self):
        return '\n'.join(self.__texts)

    @property
    def get_number_of_keys(self):
        return collection.number_of_keys


# instantiate the parser and fed it some HTML
"""
parser = MyHTMLParser()
parser.feed('<html><head><title>Test</title></head>'
            '<body><h1>Parse me!</h1></body></html>')
"""
