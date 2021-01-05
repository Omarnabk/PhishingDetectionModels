import pandas as pd
import tensorflow as tf
from keras import backend as K
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_fscore_support


def print_result(y, y_pred):
    print('macro F1 is: ', f1_score(y, y_pred, average='macro'))
    print('macro recall is: ', recall_score(y, y_pred, average='macro'))
    print('macro precision is: ', precision_score(y, y_pred, average='macro'))

    print()
    print(confusion_matrix(y, y_pred))
    print(metrics.accuracy_score(y, y_pred))
    report = classification_report(y, y_pred)
    print(report)


def classification_report_csv(y_true, y_pred, report_path):
    metrics_summary = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum()
    avg[-1] = total

    class_report_df['avg / total'] = avg
    df_class_report = class_report_df.T

    df_class_report.to_csv(report_path, sep=',')


def recall_m(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    recall = tp / (tp + fn + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    precision = tp / (tp + fp + K.epsilon())

    return precision


def f1_m(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def save_history(path, history):
    # convert the history.history dict to a pandas DataFrame:
    hist_df = pd.DataFrame(history)

    # save to json:
    hist_json_file = path + 'history.json'
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save to csv:
    hist_csv_file = path + 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
