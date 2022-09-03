from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import numpy as np
import warnings
import models
import infos

class information():
    def __init__(self):
        self.info_train = {
                'age': [],
                'workclass': [],
                'fnlwgt': [],
                'education': [],
                'education-num': [],
                'marital_status': [],
                'occupation': [],
                'relationship': [],
                'race': [],
                'sex': [],
                'capital_gain': [],
                'capital_loss': [],
                'hours-per_week': [],
                'native_country': []
        }
        self.accept_train = []

    def read_info(self, filepath):
        file = open(filepath, 'r')
        Lines = file.readlines()
        not_valid = ['workclass', 'education', 'marital_status', 'occupation',
                     'relationship', 'race', 'sex', 'native_country', 'accept']
        for line in Lines:
            per_score = line.strip().split()
            for i in range(len(per_score)):
                per_score[i] = per_score[i].replace(',', '')
                if i == len(per_score) - 1:
                    self.accept_train.append(infos.accept(per_score[i]))
                else:
                    if list(self.info_train.keys())[i] in not_valid:
                        bar = getattr(infos, list(self.info_train.keys())[i])
                        list(self.info_train.values())[i].append(bar(per_score[i]))
                    else:
                        # age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week
                        list(self.info_train.values())[i].append(int(per_score[i]))

    def dict_to_list(self):
        info_list = np.zeros((len(self.info_train), len(self.info_train['age'])), dtype=int)
        count = 0
        for i in self.info_train:
            info_list[count] = self.info_train[i]
            count += 1
        info_list = info_list.transpose()
        return info_list

    def model(self, info_list):
        X_train, X_test, y_train, y_test = train_test_split(info_list, self.accept_train, test_size=0.33, random_state=42)
        models.decision_tree(X_train, X_test, y_train, y_test)
        models.naiive_base(X_train, X_test, y_train, y_test)
        models.logistic_regression(X_train, X_test, y_train, y_test)
        models.knn(X_train, X_test, y_train, y_test)
        models.svm(X_train, X_test, y_train, y_test)
        models.mlp_sklearn(X_train, X_test, y_train, y_test)
        models.mlp_pytorch(info_list, self.accept_train)


def ignore_warning():
    warnings.filterwarnings("ignore")


def normalize(info_list):
    info_list = info_list.transpose()
    info_list = preprocessing.normalize(info_list)
    info_list = np.round(info_list, 2)
    info_list = info_list.transpose()
    return info_list


if __name__ == '__main__':
    ignore_warning()
    information = information()
    information.read_info('info.txt')
    info_list = information.dict_to_list()
    info_list = normalize(info_list)
    information.model(info_list)