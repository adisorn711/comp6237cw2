import os, sys
from AJHTMLParser import *
import json

READ_PATH = "../dataset/gap-html"
WR_PATH = "../db"

class DataReader():

    def __init__(self):
        #self.__parser = SimpleParser()
        self.__parser = BeautifulSoapParser()

    def readDataFromPath(self, path):
        f = open(path, 'r')
        if f is not None:
            html_input = f.read()
            #print "html: " + html_input
            self.__parser.feed_parser(html_input)
            f.close()
            self.reset()

    def reset(self):
        self.__parser.reset_parser()

    def reset_bag(self):
        self.__parser.clear_bag()

    @property
    def data(self):
        return self.__parser.output

    @property
    def raw_data(self):
        return self.__parser.raw_output

class DataWriter():

    def __init__(self):
        pass

    def writeDataToFile(self, json_data, path):
        with open(path, 'w+') as f:
            json.dump(json_data, f)
            f.close()

    def write_txt_to_file(self, txt, path):
        with open(path, 'w+') as f:
            f.write(txt)
            f.close()

    def create_path_if_not_exists(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def string_path_by_appending_string(s1, s2):
        return s1 + '/' + s2

