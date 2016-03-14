import os
import sys
from AJDataMining import *

READ_PATH = "../dataset/gap-html"
WR_PATH = "../db"

def main():

    root_path = READ_PATH
    dirs = os.listdir(root_path)
    output_dirs = os.listdir(WR_PATH)
    if ".DS_Store" in output_dirs:
            output_dirs.remove(".DS_Store")

    if len(output_dirs) <= 0:
        data_reader = DataReader()
        data_writer = DataWriter()
        for d in dirs:
            next_path = DataWriter.string_path_by_appending_string(root_path, d)
            files = os.listdir(next_path)

            new_dir = DataWriter.string_path_by_appending_string(WR_PATH, d)
            data_writer.create_path_if_not_exists(new_dir)

            for f_name in files:
                html_file_path = DataWriter.string_path_by_appending_string(next_path, f_name)
                data_reader.readDataFromPath(html_file_path)

            res = data_reader.data
            out_path = DataWriter.string_path_by_appending_string(new_dir, 'out.json')
            data_writer.writeDataToFile(res, out_path)
            data_reader.reset_bag()




if __name__ == '__main__':
    main()
