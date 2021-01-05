import os.path as osp
import re

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from data_utils_general import print_result

model_name = 'char'
MIN_DATASET_SIZE = 10000000000


def load_file(path):
    with open(path, encoding='utf-8') as rdr:
        lines = [r for r in rdr.readlines()]
    return lines


def feat_extractor_char(FN_train):
    vectorizer = TfidfVectorizer(ngram_range=(2, 6), analyzer='char', max_df=0.9995, lowercase=True, min_df=0.0005)
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


def _preprocess_path_sample(x):
    x = re.sub(r'[_-]', ' ', x).strip()

    # Replace numbers
    x = re.sub(r'[0-9]+', '$', x)

    # Replace special char
    x = re.sub(r'[^$0-9A-Za-z\s]+', '#', x)

    # Split of capital letter
    x = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', x)

    x = x.strip()

    return x


def load_preprocess_csv(path, train=True):
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


if __name__ == "__main__":
    print('Load training data')
    df_train = load_preprocess_csv('./dataset/train.csv', train=True)
    df_test_1 = load_preprocess_csv('./dataset/test.csv', train=False)
    df_test_2 = load_preprocess_csv('./dataset/validate.csv', train=False)
    df_test = df_test_1.append(df_test_2, ignore_index=True)

    print('Before Cleaning', len(df_train['path'].values) + len(df_test['path'].values))
    print('sus:', len(df_train[df_train.cat == 1]) + len(df_test[df_test.cat == 1]))
    print('reg:', len(df_train[df_train.cat == 0]) + len(df_test[df_test.cat == 0]))

    print('\nTraining set length:', len(df_train['path'].values))
    print('\nTesting set length:', len(df_test['path'].values))

    print('\nStart pre-processing')

    df_train.drop_duplicates(subset='path', keep='first', inplace=True)
    df_test.drop_duplicates(subset='path', keep='first', inplace=True)

    df_train['path'] = list(df_train['path'].apply(lambda x: _preprocess_path_sample(x)))
    df_test['path'] = list(df_test['path'].apply(lambda x: _preprocess_path_sample(x)))

    x_train = list(df_train['path'].values)
    x_test = list(df_test['path'].values)

    y_train = list(df_train['cat'].values)
    y_test = list(df_test['cat'].values)

    print('\nAfter Cleaning', len(x_train) + len(x_test))
    print('sus:', len(df_train[df_train.cat == 1]) + len(df_test[df_test.cat == 1]))
    print('reg:', len(df_train[df_train.cat == 0]) + len(df_test[df_test.cat == 0]))

    print('\nTraining set length:', len(x_train))
    print('\nTesting set length:', len(x_test))

    print('\nVectorizing')
    vectorizer_char = feat_extractor_char(x_train)

    print('\nTransforming')
    x_train = vectorizer_char.transform(x_train)
    x_test = vectorizer_char.transform(x_test)

    print('\nClassifying')
    lr_clf = classifier(x_train, y_train)

    print('\nPredicting')
    y_pred_test = lr_clf.predict(x_test)


    print('\nResults!!')
    print_result(y_test, y_pred_test)

    save_model(vectorizer_char, 'vect_word_path_TFIDF.pkl')
    save_model(lr_clf, 'clf_path_word_LR.pkl')

    print('DONE')
