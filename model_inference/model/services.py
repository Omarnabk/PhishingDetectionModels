#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import joblib
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from CharCNN_Models.data_utils import Data

__author__ = "Wesam Al-Nabki"


class ServiceRoutines:
    """
    A class for the services offered by FNC:
    1- FNC: a service for file name classification
    """

    def __init__(self):
        """
        Initialization function for the FNC function
        """

        classification_c_model_path = '../compiled_models/char_ngram_tfidf_model_phVSLeg/clf_FN_char_ngram_LR_5.pkl'
        vectorization_model_path_only_char = '../compiled_models/char_ngram_tfidf_model_phVSLeg/vect_ngram_fn_TFIDF_5.pkl'

        self.classifier = joblib.load(classification_c_model_path)
        self.vectorizer_char = joblib.load(vectorization_model_path_only_char)

        # kim model:
        cnn_model_root_path_kim = '../compiled_models/char_kim_model_phvsleg/'
        json_file = open(cnn_model_root_path_kim + 'model_0.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.cnn_model_kim = model_from_json(loaded_model_json)
        # load weights into new model
        self.cnn_model_kim.load_weights(cnn_model_root_path_kim + "model_0.h5")
        print("Loaded model from disk")

        # zhang model:
        cnn_model_root_path_zhang = '../compiled_models/char_zhang_model_phvsleg/'
        json_file = open(cnn_model_root_path_zhang + 'model_0.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.cnn_model_zhang = model_from_json(loaded_model_json)
        # load weights into new model
        self.cnn_model_zhang.load_weights(cnn_model_root_path_zhang + "model_0.h5")
        print("Loaded model from disk")

        cnn_model_config = json.load(open("../model_builder/CharCNN_Models/config.json"))

        self.testing_data = Data(alphabet=cnn_model_config["data"]["alphabet"],
                                 input_size=cnn_model_config["data"]["input_size"],
                                 num_of_classes=2)

    def classify_domain_name_charngramTFIDF(self, text):
        # Encode the text
        v_char = self.vectorizer_char.transform(text)

        # Predict the text category:
        fn_class_prob = self.classifier.predict_proba(v_char)
        return np.argmax(fn_class_prob, axis=1)

    def classify_domain_name_cnn_kim(self, X_test, y_test):
        testing_inputs, testing_labels = self.testing_data.get_all_data(X_test, y_test)

        y_pred_testing = self.cnn_model_kim.predict(testing_inputs, batch_size=1024, verbose=1)
        return np.argmax(y_pred_testing, axis=1)

    def classify_domain_name_cnn_zhang(self, X_test, y_test):
        testing_inputs, testing_labels = self.testing_data.get_all_data(X_test, y_test)

        y_pred_testing = self.cnn_model_zhang.predict(testing_inputs, batch_size=1024, verbose=1)
        return np.argmax(y_pred_testing, axis=1)


def load_json(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
        return data


if __name__ == "__main__":
    # the output of the classifier is:
    # class 0 --> phishing, class 1--> normal

    sr = ServiceRoutines()

    testset = load_json('../model_builder/dataset/LegitimateLogin-30K.json')
    # all the test set here is normal (legitimate) --> all "1"
    y_true = [1] * len(testset)

    y_pred = sr.classify_domain_name_charngramTFIDF(testset)

    print('\n{}'.format(classification_report(y_pred, y_true)))
    print('\n{}'.format(confusion_matrix(y_pred, y_true)))
    print('\nF1-Score: {}'.format(f1_score(y_pred, y_true)))
    print('\nAccuracy: {}'.format(accuracy_score(y_pred, y_true)))

    y_pred = sr.classify_domain_name_cnn_kim(testset, y_true)
    print('\n{}'.format(classification_report(y_pred, y_true)))
    print('\n{}'.format(confusion_matrix(y_pred, y_true)))
    print('\nF1-Score: {}'.format(f1_score(y_pred, y_true)))
    print('\nAccuracy: {}'.format(accuracy_score(y_pred, y_true)))

    y_pred = sr.classify_domain_name_cnn_zhang(testset, y_true)
    print('\n{}'.format(classification_report(y_pred, y_true)))
    print('\n{}'.format(confusion_matrix(y_pred, y_true)))
    print('\nF1-Score: {}'.format(f1_score(y_pred, y_true)))
    print('\nAccuracy: {}'.format(accuracy_score(y_pred, y_true)))
