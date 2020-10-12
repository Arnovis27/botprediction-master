import numpy as np
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


class Methods(object):
    cores = multiprocessing.cpu_count() - 1
    dict_classifiers = {
        'SVM': SVC(C=1000, kernel='poly', probability=True),
        'LogisticRegression': LogisticRegression(C=10, solver='sag', n_jobs=cores),
        'RandomForest': RandomForestClassifier(criterion='gini', n_jobs=cores)
    }
