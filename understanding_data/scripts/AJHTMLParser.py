from HTMLParser import HTMLParser
import copy
import collections

class CustomeCollection(object):

    def __init__(self):
        self.__dic = collections.OrderedDict()

    def add_object_for_key(self, key, obj):
        self.__dic[key] = obj

    def increase_value_for_key(self, key):
        if key in self.__dic:
            self.__dic[key] += 1
        else:
            self.__dic[key] = 1

    def clear_collection(self):
        self.__dic = dict()

    @property
    def bag_of_words(self):
        return copy.deepcopy(self.__dic)

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
            #print "Encountered some data  :", data
            collection.increase_value_for_key(data)

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
    def get_number_of_keys(self):
        return collection.number_of_keys

# instantiate the parser and fed it some HTML
"""
parser = MyHTMLParser()
parser.feed('<html><head><title>Test</title></head>'
            '<body><h1>Parse me!</h1></body></html>')
"""
