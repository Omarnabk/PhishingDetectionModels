import re

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle

SEED = 10


def clean_sentence(x):
    x = re.sub(r'[_-]', ' ', x).strip()

    # Replace numbers
    x = re.sub(r'[0-9]+', '$', x)

    # Replace special char
    x = re.sub(r'[^$0-9A-Za-z\s]+', '#', x)

    # Split of capital letter
    x = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', x)

    x = x.strip()

    return x


def vect(name, x_train):
    if name == 'bow':
        vec = CountVectorizer(dtype=np.float32,
                              strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                              max_df=0.9995, lowercase=True, min_df=0.0005,
                              ngram_range=(1, 3)).fit(x_train)

    elif name == 'tfidf':
        vec = TfidfVectorizer(dtype=np.float32, min_df=3, max_features=None,
                              strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                              ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                              stop_words='english').fit(x_train)

    return vec


def classifier(x_train, y_train, name, x_validate=None, y_validate=None):
    if name == 'lr':
        clf = LogisticRegression(random_state=0,
                                 solver='liblinear',
                                 C=100, max_iter=1000,
                                 class_weight='balanced').fit(x_train, y_train)
    elif name == 'nb':
        clf = MultinomialNB().fit(x_train, y_train)

    return clf


# small function to find threshold and find best f score - Eval metric of competition
def print_result(y, y_pred):
    print('macro F1 is: ', f1_score(y, y_pred, average='macro'))
    print('macro recall is: ', recall_score(y, y_pred, average='macro'))
    print('macro precision is: ', precision_score(y, y_pred, average='macro'))

    print()
    print(confusion_matrix(y, y_pred))
    print(metrics.accuracy_score(y, y_pred))


def save_model(model, model_name):
    joblib.dump(model, model_name)


if __name__ == "__main__":
    train_df = pd.read_csv("./dataset/train.csv")
    train_df = shuffle(train_df)
    test_df = pd.read_csv("./dataset/test.csv")
    valid_df = pd.read_csv("./dataset/validate.csv")
    print("Train shape : ", train_df.shape)
    print("Validate shape : ", valid_df.shape)
    print("Test shape : ", test_df.shape)

    train_df['cleaned_text'] = train_df['path'].apply(lambda x: clean_sentence(x))
    valid_df['cleaned_text'] = valid_df['path'].apply(lambda x: clean_sentence(x))
    test_df['cleaned_text'] = test_df['path'].apply(lambda x: clean_sentence(x))

    print("Train shape : ", len(train_df.cleaned_text))
    print("Validate shape : ", len(valid_df.cleaned_text))
    print("Test shape : ", len(test_df.cleaned_text))

    # Activate only when release the model
    train_df = train_df.append(valid_df, ignore_index=True).append(test_df, ignore_index=True)

    vectorizer_path = vect(name='tfidf', x_train=train_df.cleaned_text.values)

    x_train = vectorizer_path.transform(train_df.cleaned_text.values)
    x_test = vectorizer_path.transform(test_df.cleaned_text.values)
    x_validate = vectorizer_path.transform(valid_df.cleaned_text.values)

    y_train = train_df.cat.values
    y_test = test_df.cat.values
    y_validate = valid_df.cat.values

    clf_path = classifier(x_train, y_train, name='lr')

    save_model(vectorizer_path, './model_tfidf_word_path/vect_word_path_TFIDF.pkl')
    save_model(clf_path, './model_tfidf_word_path/clf_path_word_LR.pkl')

    y_valid_pred = clf_path.predict(x_validate)
    y_test_pred = clf_path.predict(x_test)

    print('Validation')
    print_result(y_validate, y_valid_pred)
    print()
    print('Testing')
    print_result(y_test, y_test_pred)
