import re

import joblib
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score

model_name = 'char'


# small function to find threshold and find best f score - Eval metric of competition
def print_result(y, y_pred):
    print('macro F1 is: ', f1_score(y, y_pred, average='macro'))
    print('macro recall is: ', recall_score(y, y_pred, average='macro'))
    print('macro precision is: ', precision_score(y, y_pred, average='macro'))

    print()
    print(confusion_matrix(y, y_pred))
    print(metrics.accuracy_score(y, y_pred))

    print(classification_report(y, y_pred))


# model_name = 'char_word'

def load_file(path):
    with open(path, encoding='utf-8') as rdr:
        lines = [r for r in rdr.readlines()]
    return lines


def clean_list(lst):
    lines = [clean_sample(x).strip() for x in lst]

    lines = list(set(lines))
    if '' in lines:
        lines.remove('')
    return lines


def clean_sample(text):
    if '.' in text:
        text = '.'.join(text.split('.')[:-1])

    return text

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
    # seed = 10
    # random.seed = seed
    #
    # sus_lines = list(set(load_file(path='./dataset_filename/sus.txt')))
    # reg_lines = list(set(load_file(path='./dataset_filename/reg.txt')[:1000000]))
    #
    # print('Before Cleaning', len(sus_lines) + len(reg_lines))
    #
    #
    # Y = [1] * len(sus_lines) + [0] * len(reg_lines)
    # X = sus_lines + reg_lines
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

    df_train = pd.read_csv('./dataset_filename/train.csv')
    df_test = pd.read_csv('./dataset_filename/test.csv')

    x_train = df_train['path'].values
    x_test = df_test['path'].values

    y_train = df_train['cat'].values
    y_test = df_test['cat'].values

    print('Before Cleaning', len(x_train) + len(x_test))
    print('sus:', len(df_train[df_train.cat == 1]) + len(df_test[df_test.cat == 1]))
    print('reg:', len(df_train[df_train.cat == 0]) + len(df_test[df_test.cat == 0]))

    print('\nTraining set length:', len(df_train['path'].values))
    print('\nTesting set length:', len(df_test['path'].values))

    print('\nStart pre-processing')
    x_train_clean = [clean_sample(x).strip() for x in x_train]
    x_test_clean = [clean_sample(x).strip() for x in x_test]

    df_train = pd.DataFrame({'path': x_train_clean, 'cat': y_train})
    df_test = pd.DataFrame({'path': x_test_clean, 'cat': y_test})

    df_train.drop_duplicates(subset='path', keep='first', inplace=True)
    df_test.drop_duplicates(subset='path', keep='first', inplace=True)

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
    y_pred = lr_clf.predict(x_test)

    print('\nResults!!')
    print_result(y_test, y_pred)

    # save_model(vectorizer_char, './Char_TFIDF_Ngram_Model/vect_ngram_fn_TFIDF.pkl')
    # save_model(lr_clf, './Char_TFIDF_Ngram_Model/clf_FN_char_ngram_LR.pkl')

    print('DONE')
