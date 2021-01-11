# model_architecture_name = 'char_ngram_tfidf_model'
model_architecture_name = 'char_zhang_Login__model'
base_mode_path = f'../compiled_models/{model_architecture_name}/'

import os

import numpy as np

avg_p, avg_r, avg_f1 = [], [], []
output = []
for x in os.listdir(base_mode_path):
    if not x.endswith('.csv'):
        continue
    with open(base_mode_path + x, 'r') as rdr:
        output.append([y.strip() for y in rdr.readlines()[-1].split(',')[1:-1]])

avg_p, avg_r, avg_f1 = list(zip(*output))
print('avg_precision', round(np.average([eval(v) for v in avg_p]), 3))
print('avg_recall', round(np.average([eval(v) for v in avg_r]), 3))
print('avg_f1', round(np.average([eval(v) for v in avg_f1]), 3))
