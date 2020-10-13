import csv
import os
import sys
import pickle
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from logic.feature_extraction import FeatureExtraction
from logic.iteration_cv import IterationCV
from sklearn.model_selection import train_test_split
from utils.utils import Util

DIR_OUTPUT= ("../data/output/")
DIR_MODELS= ("../data/models/")
DIR_INPUT= ("../data/input/es/")

fieldnames = ('model_name', 'classifier_name', 'f1', 'accuracy', 'recall', 'precision', 'cross_entropy',
              'log_loss', 'classification', 'confusion', 'sample_train', 'classifier', 'time_processing')

list_model = {'lexical': '00'}


class MachineLearning(object):
    """
    :Date: 2019-10-03
    :Version: 0.1
    :Author: Edwin Puertas
    :Copyright: GPL
    """
    def __init__(self):
        """
        :rtype: object
        :return: Machine learning object
        """
        try:
            print('Load Machine Learning....')
            self.fex = FeatureExtraction(text_analysis=None, lang='es')
        except Exception as e:
            Util.standard_error(sys.exc_info())
            print('Error constructor: {0}'.format(e))

    def train(self, file_output='predictive_sentiment', iteration=10, fold=10):
        try:
            result = {}
            best_model = None
            best_classifier = None
            best_f1 = 0.0
            date_file = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            file_report = '{0}_Fold{1}_Iteration{2}_report_{3}.csv'.format(file_output, fold, iteration, date_file)
            output = DIR_OUTPUT + file_report
            label = preprocessing.LabelEncoder()
            x_train, x_test, y_train, y_test = Util.import_dataset()

            with open(output, 'w') as out_csv:
                writer = csv.DictWriter(out_csv, fieldnames=fieldnames, delimiter=';', lineterminator='\n')
                headers = dict((n, n) for n in fieldnames)
                writer.writerow(headers)

                for model_name, value in list_model.items():
                    print('{0}| Start Model: {1}|{0}'.format("#" * 15, model_name))
                    # data train
                    print('Get train features')
                    x_train = [self.fex.get_features(text=text, model_type=value) for text in tqdm(x_train)]
                    x_train = preprocessing.normalize(x_train)
                    y_train = label.fit_transform(y_train)

                    # data test
                    print('Get test features')
                    x_test = [self.fex.get_features(text=text, model_type=value) for text in tqdm(x_test)]
                    x_test = preprocessing.normalize(x_test)
                    y_test = label.fit_transform(y_test)

                    # crear una función que reciba por parametro el modelo(algoritmo de clasificación)
                    # x_train, y_train, x_test, y_test
                    data_result = {}

                    [writer.writerow(model_i) for model_i in data_result]
                    out_csv.flush()
                    print('Model {0} save successful!'.format(model_name))

                    for row in data_result:
                        f1_j = float(row['f1'])
                        classifier = row['classifier']
                        if f1_j > best_f1:
                            best_f1 = f1_j
                            best_model = row['model_name']
                            best_classifier = row['classifier_name']
                            # save model
                            file_model = '{0}{1}_model.sav'.format(DIR_MODELS, file_output)
                            outfile = open(file_model, 'wb')
                            pickle.dump(classifier, outfile)
                            outfile.close()
                            print('Model exported in {0}'.format(file_model))
                out_csv.close()
                print('{0}| End Model: {1}|{0}'.format("#" * 15, model_name))
            print('The best model is {0}, {1} with F1 score = {2}'.format(best_model, best_classifier, best_f1))
        except Exception as e:
            Util.standard_error(sys.exc_info())
            print('Error train: {0}'.format(e))
            return None


if __name__ == "__main__":
    file_output = 'PredictiveBot'
    ml = MachineLearning()
    ml.train(file_output=file_output, fold=10, iteration=10)
