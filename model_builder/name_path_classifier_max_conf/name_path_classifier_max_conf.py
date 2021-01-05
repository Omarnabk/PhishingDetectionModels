#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import os.path as osp
import re

import joblib
import numpy as np
import pandas as pd

__author__ = "Wesam Al-Nabki"

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


def print_result(y, y_pred):
    print('macro F1 is: ', f1_score(y, y_pred, average='macro'))
    print('macro recall is: ', recall_score(y, y_pred, average='macro'))
    print('macro precision is: ', precision_score(y, y_pred, average='macro'))

    print()
    print(confusion_matrix(y, y_pred))
    print(metrics.accuracy_score(y, y_pred))

    print(classification_report(y, y_pred))


def preprocess_path(x):
    x = ' '.join(os.path.normpath(x).split(os.sep))

    x = re.sub(r'[_-]', ' ', x).strip()

    # Replace numbers
    x = re.sub(r'[0-9]+', '$', x)

    # Replace special char
    x = re.sub(r'[^$0-9A-Za-z\s]+', '#', x)

    # Split of capital letter
    x = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', x)

    x = x.strip()

    return x


def preprocess_filename(text):
    """
    Function to clean the file name ( pre-processor).
    :param text: the file name
    :return: the text preprocessed
    """

    if '.' in text:
        text = '.'.join(text.split('.')[:-1])

    ortho_code_num = re.sub(r'[^A-Za-z]+', '0', text)
    ortho_code_num = re.sub(r'[a-zA-Z]+', '1', ortho_code_num)

    ortho_code = re.sub(r'[A-Z]', 'C', text)
    ortho_code = re.sub(r'[a-z]', 'c', ortho_code)
    ortho_code = re.sub(r'[0-9]', 'N', ortho_code)
    ortho_code = re.sub(r'[^a-zA-Z0-9]', 'P', ortho_code)

    clean_text = re.sub(r'[0-9]+', '$', text)
    clean_text = re.sub(r'[^$0-9A-Za-z\s]+', '#', clean_text)
    clean_text = re.sub(r'(?:^| )\w(?:$| )', '', clean_text)
    clean_text = clean_text.strip()

    return clean_text + ' ' + ortho_code_num + ' ' + ortho_code


MIN_DATASET_SIZE = 1000000000


class ServiceRoutines:
    """
    A class for the services offered by FNC:
    1- FNC: a service for file name classification
    """

    def __init__(self):
        """
        Initialization function for the FNC function
        """
        self.suspicious_threshold = 0.7

        fn_clf_model_path = './data/clf_FN_char_ngram_LR.pkl'
        fn_vectorization_model_path = './data/vect_ngram_fn_TFIDF.pkl'

        path_clf_model_path = './data/clf_path_word_LR.pkl'
        path_vectorization_model_path = './data/vect_word_path_TFIDF.pkl'

        self.vectorizer_fn = joblib.load(fn_vectorization_model_path)
        self.classifier_fn = joblib.load(fn_clf_model_path)

        self.vectorizer_path = joblib.load(path_vectorization_model_path)
        self.classifier_path = joblib.load(path_clf_model_path)

    def classify_file_name_charngramTFIDF(self, file_path):
        """
        Function to classify input path into Safe or CSA-related. The function divides the input text into filename and
         path. Then, for each part it calls a standalone classification models, i.e. one for the filename and one for
         the path.

        :type file_path: string
        :param file_path: file name and its path to be classified

        :return: the class of the input path, as a binary flag: [1] if CSA-related file, [0] otherwise. Also, it returns
        the prediction confidence.

        :Example:
            *   (0,0.88): this output refers to a non-CSA file name with a prediction confidence of 0.88.
            *   (1,0.96): this output refers to a CSA file name with a prediction confidence of 0.96.
        """

        try:
            fn_parts = osp.split(file_path)
            path = fn_parts[0]
            filename = fn_parts[1]
        except:
            filename = file_path
            path = file_path

        # class the file name classifier
        fn_result = self.classify_filename_charngramTFIDFLR(filename=filename)

        # class the path classifier
        path_result = self.classify_path_TFIDFLR(path)

        # select the maximum suspicious confidence
        if fn_result[1] > path_result[1]:
            class_pred_prob = fn_result
        else:
            class_pred_prob = path_result

        if class_pred_prob[1] >= self.suspicious_threshold:
            return 1, round(class_pred_prob[1], 2)
        return 0, round(class_pred_prob[0], 2)

    def classify_path_TFIDFLR(self, path):
        """
        Method to classify a given path of a file.
        The model used TFIDF for features extraction and Logistic Regresion for classification.
        The function has three steps: text-preprocessing, vectorization, and classification
        The method return prediction probability of an input path

        :type path: string
        :param path: file path to be classified
        :return: the prediction probability of the given file.
            The output is a list of two items. Index zero probability of the safe class and index one is the probability
             of the suspicious class
            If the text was empty, the output will be [1.0, 0.0]

        :Example:
            *   [0.44, 0.56]: Safe class in index zero and CSA class in index 1
        """

        if not path:
            return [1.0, 0.0]
        parts = osp.normpath(path).split(osp.sep)
        path = '/'.join(parts[:-1]) + '/'
        clean_example = preprocess_path(path)

        v_path = self.vectorizer_path.transform([clean_example])

        return self.classifier_path.predict_proba(v_path)[0]

    def classify_filename_charngramTFIDFLR(self, filename):
        """
        Method to classify a filename in a given path.
        The model used char n-gram TFIDF for features extraction and Logistic Regresion for classification.
        The function has three steps: text-preprocessing, vectorization, and classification
        The method return prediction probability of an input file name

        :type filename: string
        :param filename: file name to be classified
        :return: the prediction probability of the given file.
            The output is a list of two items. Index zero probability of the safe class and index one is the probability
             of the suspicious class
            If the text was empty, the output will be [1.0, 0.0]

        :Example:
            *   [0.44, 0.56]: Safe class in index zero and CSA class in index 1
        """

        if not filename:
            return [1.0, 0.0]

        filename = str(osp.normpath(filename).split(osp.sep)[-1]).strip()

        clean_example = preprocess_filename(filename)

        v_char = self.vectorizer_fn.transform([clean_example])

        return self.classifier_fn.predict_proba(v_char)[0]


def load_preprocess_csv(path, train=False):
    def remove_file_name_from_full_path(path): return '/'.join(osp.split(path)[:-1])

    df = pd.read_csv(path, sep=',', index_col=False)
    df = shuffle(df, random_state=10)
    # To load balanced dataset for training.
    if train:
        d_pos = shuffle(df[df.cat == 1], random_state=10)
        d_neg = shuffle(df[df.cat == 0], random_state=10)
        m = min(len(d_pos), len(d_neg), int(MIN_DATASET_SIZE / 2))

        d_pos = d_pos[:m]
        d_neg = d_neg[:m]
        df = shuffle(d_pos.append(d_neg, ignore_index=True))
    df['path'] = list(df['path'].apply(lambda x: remove_file_name_from_full_path(x)))
    return df


if __name__ == '__main__':

    # TEST_SET_SIZE = 100000
    #
    # print('Load test set:')
    # df_test_fn = pd.read_csv('../dataset_file_path/test_fn.csv')
    # df_test_fn.drop_duplicates(subset='path', keep='first', inplace=True)
    #
    # df_test_1 = load_preprocess_csv('../dataset_file_path/test1_path.csv')
    # df_test_2 = load_preprocess_csv('../dataset_file_path/test2_path.csv')
    # df_test_path = df_test_1.append(df_test_2, ignore_index=True)
    # df_test_path.drop_duplicates(subset='path', keep='first', inplace=True)
    #
    # df_test_path = df_test_path.sample(n=TEST_SET_SIZE, random_state=10)
    # df_test_fn = df_test_fn.sample(n=TEST_SET_SIZE, random_state=10)
    #
    # print('Path Test size:', len(df_test_fn))
    # print('Name Test size:', len(df_test_path))
    #
    # samples, cats = [], []
    # for x1, y1, x2, y2 in zip(df_test_path['path'].values,
    #                           df_test_path['cat'].values,
    #                           df_test_fn['path'].values,
    #                           df_test_fn['cat'].values):
    #     samples.append(osp.join(str(x1.strip()), str(x2.strip())))
    #     cat = 0
    #     if y1 == 1 or y2 == 1:
    #         cat = 1
    #     cats.append(cat)
    #
    # df_test = pd.DataFrame({'path': samples, 'cat': cats})
    # df_test.to_csv('../dataset_filename_filepath/test_fn_path_randomseed_10.csv')

    # Load Pedro file:
    import pandas as pd
    root = 'D:/Google_Drive/INCIBE_Deliveries_Documentations/FileNameClassification/test samples/'
    pos = root + 'result_HD.txt'
    neg = root + 'Andres_d.txt'

    with open(pos, 'r', encoding='utf-8') as rdr1:
        pos_s = [x.strip() for x in rdr1.readlines()]
        pos_l = [1] * len(pos_s)

    with open(neg, 'r', encoding='utf-8') as rdr2:
        neg_s = [x.strip() for x in rdr2.readlines()]
        neg_l = [0] * len(neg_s)

    pos_df = pd.DataFrame({'path': pos_s, 'cat': pos_l})
    neg_df = pd.DataFrame({'path': neg_s, 'cat': neg_l})

    pos_df_sub = pos_df.sample(n=25000)
    neg_df_sub = neg_df.sample(n=25000)

    from sklearn.utils import shuffle
    seed = 10
    df = shuffle(pos_df_sub.append(neg_df_sub, ignore_index=True), random_state=seed)
    df.to_csv('../dataset_filename_filepath/dataframe_test_dup.csv', index=False)

    # Load Pedro file:

    #
    # root = 'D:/Google_Drive/INCIBE_Deliveries_Documentations/FileNameClassification/test samples/'
    # df_test_1 = load_preprocess_csv('../dataset/test.csv')
    # df_test_2 = load_preprocess_csv('../dataset/validate.csv')
    # df_test = df_test_1.append(df_test_2, ignore_index=True)
    #
    # pos_s = list(df_test[df_test.cat == 1].values)
    # pos_l = [1] * len(pos_s)
    # neg = root + 'Andres_d.txt'
    #
    # with open(neg, 'r', encoding='utf-8') as rdr2:
    #     neg_s = [x.strip() for x in rdr2.readlines()]
    #     neg_l = [0] * len(neg_s)
    #
    # pos_df = pd.DataFrame({'path': pos_s, 'cat': pos_l})
    # neg_df = pd.DataFrame({'path': neg_s, 'cat': neg_l})
    #
    # pos_df_sub = pos_df.sample(n=25000)
    # neg_df_sub = neg_df.sample(n=25000)
    #
    # from sklearn.utils import shuffle
    #
    # seed = 10
    # df = shuffle(pos_df_sub.append(neg_df_sub, ignore_index=True), random_state=seed)
    # df.to_csv('../dataset_filename_filepath/dataframe_test.csv', index=False)

    df_test = pd.read_csv('../dataset_filename_filepath/dataframe_test.csv')

    print('sus:', len(df_test[df_test.cat == 1]))
    print('reg:', len(df_test[df_test.cat == 0]))

    fnc_service = ServiceRoutines()
    gt_list, pred_list_m1, pred_list_m2 = [], [], []

    for sample, cat in zip(df_test['path'].values, df_test['cat'].values):
        gt_list.append(cat)

        # Method 1: use two classifiers
        ##########################
        pred = fnc_service.classify_file_name_charngramTFIDF(sample)[0]
        pred_list_m1.append(pred)
        ##########################

        ## Method 2: split the path and use only the file name classifier
        #########################
        path_parts = sample.split('\\')
        p = 0
        for sample_subdir in path_parts:
            pred = int(np.argmax(fnc_service.classify_filename_charngramTFIDFLR(filename=str(sample_subdir))))
            if pred == 1:
                p = 1
                break
        pred_list_m2.append(p)
        #########################

    print('\nResults method_1!!')
    print_result(gt_list, pred_list_m1)

    print('\nResults method_2!!')
    print_result(gt_list, pred_list_m2)
