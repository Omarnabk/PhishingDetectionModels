#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path as osp

import joblib

from ..model.resources import clean_sample_char

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

        this_dir = osp.dirname(__file__)
        model_path = osp.join(this_dir, '..')

        classification_c_model_path = model_path + '/data/clf_c.pkl'
        vectorization_model_path_only_char = model_path + '/data/vect_only_char.pkl'

        self.FNC_Threshold = 0.70
        self.classifier = joblib.load(classification_c_model_path)
        self.vectorizer_char = joblib.load(vectorization_model_path_only_char)

    def classify_file_name_charngramTFIDF(self, text):
        """
        File Name Classification method based on TFIDF N-gram for text representation and Logistic Regression for Text
        classification
    
        :type text: string
        :param text: file name to be classified it is CSA related or not
    
        :return: the class of the file name as a binary flag: [1] if CSA-related file, [0] otherwise. Also, it returns
        the prediction confidence.
    
        :Example:
            *   (0,0.88): this output refers to a non-CSA file name with a prediction confidence of 0.88.
            *   (1,0.96): this output refers to a CSA file name with a prediction confidence of 0.96.
        """

        clean_example = clean_sample_char(text)

        # Encode the text
        v_char = self.vectorizer_char.transform([clean_example])

        if clean_example:
            # Predict the text category:
            fn_class_prob = self.classifier.predict_proba(v_char)[0]
        else:
            return -1, 0.0

        if fn_class_prob[1] >= self.FNC_Threshold:
            return 1, round(fn_class_prob[1], 2)
        return 0, round(fn_class_prob[0], 2)
