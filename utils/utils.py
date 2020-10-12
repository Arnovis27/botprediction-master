import csv
import hashlib
import re
import sys
import os
import os.path
import datetime
import unicodedata
from datetime import datetime
from xml.dom import minidom
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from root import DIR_INPUT


class Util(object):
    """
    :Date: 2019-10-03
    :Version: 0.1
    :Author: Gabriel Moreno & Edwin Puertas - Pontificia Universidad Javeriana, Bogotá
    :Copyright: To be defined
    :Organization: Centro de Excelencia y Apropiación de Big Data y Data Analytics - CAOBA
    This class has static methods
    """
    def __init__(self):
        print("Class Utils")

    @staticmethod
    def sha1(text):
        result = hashlib.sha1((text + str(datetime.datetime.now())).encode())
        return result.hexdigest()

    @staticmethod
    def proper_encoding(text):
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        return text

    @staticmethod
    def walklevel(some_dir, level=1):
        some_dir = some_dir.rstrip(os.path.sep)
        assert os.path.isdir(some_dir)
        num_sep = some_dir.count(os.path.sep)
        for root, dirs, files in os.walk(some_dir):
            yield root, dirs, files
            num_sep_this = root.count(os.path.sep)
            if num_sep + level <= num_sep_this:
                del dirs[:]

    @staticmethod
    def standard_error(error_data):
        try:
            exc_type, exc_obj, exc_tb = error_data
            return \
                'ERROR: ' + exc_type.__name__ + ': ' + str(exc_obj) + '\nFILE: ' + exc_tb.tb_frame.f_code.co_filename + \
                '\nMETHOD: ' + exc_tb.tb_frame.f_code.co_name + \
                '\nLINE: ' + str(exc_tb.tb_lineno) + \
                '\n------------------------------------------------------------------------'
        except Exception:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            return \
                'ERROR: ' + exc_type.__name__ + ': ' + str(exc_obj) + '\nFILE: ' + exc_tb.tb_frame.f_code.co_filename + \
                '\nMETHOD: ' + exc_tb.tb_frame.f_code.co_name + \
                '\nLINE: ' + str(exc_tb.tb_lineno) + \
                '\n------------------------------------------------------------------------'

    @staticmethod
    def import_dataset():
        try:
            count_file = 0
            dict_tweets = {}
            data = []
            if os.path.exists(DIR_INPUT):
                print("\t** #:READ in path {0}.".format(DIR_INPUT))
                for subdir, dirs, files in os.walk(DIR_INPUT):
                    for file in tqdm(files):
                        count_file += 1
                        file_path = DIR_INPUT + file
                        name_user = file.replace('.xml', '')
                        if '.xml' in file_path:
                            file_author = minidom.parse(file_path)
                            docs = file_author.getElementsByTagName('document')
                            tweets = ''
                            for doc in docs:
                                text_local = re.sub(r"\s+", " ", doc.firstChild.data)
                                tweets += text_local + '\n'
                            dict_tweets[name_user] = tweets

                        if '.txt' in file_path:
                            # Load Data
                            with open(file_path, newline='', encoding='UTF-8') as csv_chat:
                                reader = csv.reader(csv_chat)
                                # title = True
                                count = 0
                                for row_string in reader:
                                    count += 1
                                    row = str(row_string[0]).replace(':::', ';').split(';')
                                    data.append([row[0], dict_tweets[row[0]], row[1]])
            x = [i[1] for i in data]
            y = [i[2] for i in data]
            return train_test_split(x, y, test_size=0.30, random_state=8675309)
        except Exception as e:
            Util.standard_error(sys.exc_info())
            print('Error import_dataset: {0}'.format(e))