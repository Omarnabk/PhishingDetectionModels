import re

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def _preprocess_sample(x):
    x = re.sub(r'[_-]', ' ', x).strip()

    # Replace numbers
    x = re.sub(r'[0-9]+', '$', x)

    # Replace special char
    x = re.sub(r'[^$0-9A-Za-z\s]+', '#', x)

    # Split of capital letter
    x = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', x)

    x = x.strip()

    return x


def load_csv(path):
    df = pd.read_csv(path, sep=',', index_col=False)
    df = shuffle(df, random_state=10)
    d_pos = shuffle(df[df.cat == 1], random_state=10)
    d_neg = shuffle(df[df.cat == 0], random_state=10)
    m = min(len(d_pos), len(d_neg), 10000000000000)

    d_pos = d_pos[:m]
    d_neg = d_neg[:m]
    df = shuffle(d_pos.append(d_neg, ignore_index=True))
    paths = list(df['path'].apply(lambda x: _preprocess_sample(x)))
    cats = list(df['cat'].values)

    items = list(zip(paths, cats))
    np.random.shuffle(items)
    paths, cats = zip(*items)
    return np.array(paths), np.array(cats)


class Data(object):
    """
    Class to handle loading and processing of raw datasets.
    """

    def __init__(self,
                 alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 input_size=1014, num_of_classes=4):
        """
        Initialization of a Data object.

        Args:
            data_source (str): Raw data file path
            alphabet (str): Alphabet of characters to index
            input_size (int): Size of input features
            num_of_classes (int): Number of classes in data
        """
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}  # Maps each character to an integer
        self.no_of_classes = num_of_classes
        for idx, char in enumerate(self.alphabet):
            self.dict[char] = idx + 1
        self.length = input_size

    def get_all_data(self, x, y):
        """
        Return all loaded data from data variable.

        Returns:
            (np.ndarray) Data transformed from raw to indexed form with associated one-hot label.

        """
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for x, y in zip(x, y):
            batch_indices.append(self.str_to_indexes(x))
            classes.append(one_hot[int(y)])
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def str_to_indexes(self, s):
        """
        Convert a string to character indexes based on character dictionary.
        
        Args:
            s (str): String to be converted to indexes

        Returns:
            str2idx (np.ndarray): Indexes of characters in s

        """
        s = s.lower()
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
        return str2idx
