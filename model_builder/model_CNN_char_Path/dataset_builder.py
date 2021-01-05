import os
import json
from sklearn.model_selection import train_test_split
import pandas as pd

DATASET_ROOT_PATH = '../../model_builder/dataset/'


def load_json(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
        return data


def write2csv(file_name, all_x, all_y):
    with open(file_name, 'w', encoding='utf-8') as wrt:
        for x, y in zip(all_x, all_y):
            wrt.write('{}\t{}\n'.format(x, y))


x_class_0 = load_json(DATASET_ROOT_PATH + 'Phishing-20K.json')
y_class_0 = len(x_class_0) * [0]
x_class_1 = load_json(DATASET_ROOT_PATH + 'LegitimateLogin-20K.json')
y_class_1 = len(x_class_1) * [1]
x_class_2 = load_json(DATASET_ROOT_PATH + 'Legitimate-20K.json')
y_class_2 = len(x_class_2) * [2]

X = x_class_0 + x_class_1
y = y_class_0 + y_class_1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

df_train = pd.DataFrame({'domain': X_train, 'class': y_train})
df_test = pd.DataFrame({'domain': X_test, 'class': y_test})
df_dev = pd.DataFrame({'domain': X_val, 'class': y_val})

df_train.to_csv(DATASET_ROOT_PATH + 'train.csv', sep=';', index=False)
df_dev.to_csv(DATASET_ROOT_PATH + 'dev.csv', sep=';', index=False)
df_test.to_csv(DATASET_ROOT_PATH + 'test.csv', sep=';', index=False)
print(done)
