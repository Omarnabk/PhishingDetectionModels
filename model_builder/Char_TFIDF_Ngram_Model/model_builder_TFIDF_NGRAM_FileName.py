import json
import os
import re

import joblib
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from data_utils_general import classification_report_csv

model_name = 'char'


def write2csv(file_name, all_x, all_y):
    with open(file_name, 'w', encoding='utf-8') as wrt:
        for x, y in zip(all_x, all_y):
            wrt.write('{}\t{}\n'.format(x, y))


def load_json(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
        return data


# small function to find threshold and find best f score - Eval metric of competition
def print_result(y, y_pred):
    print('macro F1 is: ', f1_score(y, y_pred, average='macro'))
    print('macro recall is: ', recall_score(y, y_pred, average='macro'))
    print('macro precision is: ', precision_score(y, y_pred, average='macro'))

    print()
    print(confusion_matrix(y, y_pred))
    print(metrics.accuracy_score(y, y_pred))

    print(classification_report(y, y_pred))


def feat_extractor_char(FN_train):
    vectorizer = TfidfVectorizer(ngram_range=(2, 6), analyzer='char', max_df=0.9999, lowercase=True, min_df=0.0005)
    vectorizer = vectorizer.fit(FN_train)
    return vectorizer


def classifier(x_train, y_train):
    lr_clf = LogisticRegression(random_state=0,
                                solver='liblinear',
                                C=100, max_iter=1000,
                                class_weight='balanced').fit(x_train, y_train)
    return lr_clf


def save_model(model, model_name):
    joblib.dump(model, model_name)


def df_voc(df_p, cat):
    df_p = df_p[df_p.cat == cat]
    all_v = [x.split() for x in df_p.path.values]
    all_v = list(set([item for sublist in all_v for item in sublist
                      if 4 < len(re.sub(r'[^a-zA-Z]', '', item)) < 10]))
    return all_v


if __name__ == "__main__":
    DATASET_ROOT_PATH = '../model_builder/dataset/'

    # Phishing VS Legitimate
    # model_architecture_name = 'char_ngram_tfidf_model_phVSLeg'

    # Phishing VS LegitimateLogin
    model_architecture_name = 'char_ngram_tfidf_model_phVSLegLogin'

    base_mode_path = f'../compiled_models/{model_architecture_name}/'

    if not os.path.exists(base_mode_path):
        os.makedirs(base_mode_path)

    x_class_ph = load_json(DATASET_ROOT_PATH + 'Phishing-30K.json')
    y_class_ph = len(x_class_ph) * [0]

    x_class_lg = load_json(DATASET_ROOT_PATH + 'Legitimate-30K.json')
    y_x_class_lg = len(x_class_lg) * [1]

    x_class_lg_login = load_json(DATASET_ROOT_PATH + 'LegitimateLogin-30K.json')
    y_x_class_lg_login = len(x_class_lg_login) * [2]

    # Phishing VS Legitimate
    # X = x_class_ph + x_class_lg
    # y = y_class_ph + y_x_class_lg

    # Phishing VS LegitimateLogin
    X = x_class_ph + x_class_lg_login
    y = y_class_ph + y_x_class_lg_login

    X, y = shuffle(X, y, random_state=42)

    X = np.array(X)
    y = np.array(y)

    X, _, y, _ = train_test_split(X, y, test_size=0.0, random_state=42)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold_id, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        print('\nVectorizing')
        vectorizer_char = feat_extractor_char(X_train)

        print('\nTransforming')
        x_train = vectorizer_char.transform(X_train)
        x_test = vectorizer_char.transform(X_test)

        print('\nClassifying')
        lr_clf = classifier(x_train, y_train)

        print('\nPredicting')
        y_pred = lr_clf.predict(x_test)

        print('\nResults!!')

        report_name = base_mode_path + f'clf_{fold_id}.csv'
        print_result(y_test, y_pred)
        classification_report_csv(y_test, y_pred, report_name)

        save_model(vectorizer_char, base_mode_path + f'/vect_ngram_fn_TFIDF_{fold_id}.pkl')
        save_model(lr_clf, base_mode_path + f'clf_FN_char_ngram_LR_{fold_id}.pkl')
        print('DONE')
