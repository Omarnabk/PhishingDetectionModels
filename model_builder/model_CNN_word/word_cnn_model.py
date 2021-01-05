import random

from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import *
from keras.models import *
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tqdm.auto import tqdm

from model_CNN_char_Path.data_utils import *

tqdm.pandas(desc='Progress')

import os

# cross validation and metrics

import numpy as np

embed_size = 300  # how big is each word vector
max_features = 120000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70  # max number of words in a question to use
batch_size = 512  # how many samples to process at once
n_epochs = 5  # how many times to iterate over all samples
n_splits = 5  # Number of K-fold Splits
SEED = 10
debug = 0


def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def clean_sentence(x):
    x = re.sub(r'(\b\w\b)|[\\/]', ' ', x).strip()
    x = re.sub(r'([^A-Za-z0-9]\s)', ' ', x).strip()
    x = re.sub(r'[0-9]+', '$', x)
    x = re.sub(r'[^$0-9A-Za-z\s]+', '#', x)
    x = ' '.join(re.findall('[A-Z][^A-Z]*', x))
    return x.strip().lower()


def load_glove(word_index):
    EMBEDDING_FILE = 'D:/GDrive/Work/PyCharmProjects/Advanced TC/embeddings/glove.840B.300d/glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')[:300]

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8'))
    in_l = 0
    out_l = 0
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = -0.005838499, 0.48782197
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        # ALLmight
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            in_l += 1
        else:
            embedding_vector = embeddings_index.get(word.capitalize())
            if embedding_vector is not None:
                in_l += 1
                embedding_matrix[i] = embedding_vector
            else:
                out_l += 1
    print(in_l, '    ', out_l)
    return embedding_matrix


# https://www.kaggle.com/yekenot/2dcnn-textclassifier
def model_cnn(embedding_matrix):
    filter_sizes = [1, 2, 3, 5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(min(max_features, embedding_matrix.shape[0]), embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(filters=num_filters,
                      kernel_size=(filter_sizes[i], embed_size),
                      activation='relu',
                      kernel_initializer='he_normal'
                      )(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z = Dropout(0.5)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_m, recall_m, precision_m])

    return model


def trainMe(model, training_inputs, training_labels, validation_inputs,
            validation_labels, epochs, batch_size, checkpoint_every=100):
    tensorboard = TensorBoard(log_dir='.', histogram_freq=checkpoint_every, batch_size=batch_size,
                              write_graph=False, write_grads=True, write_images=False,
                              embeddings_freq=checkpoint_every,
                              embeddings_layer_names=None
                              )

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=5,
                                   verbose=1,
                                   mode='min')

    # Start training
    print("Training model: ")
    history = model.fit(training_inputs, training_labels,
                        # validation_split=0.1,
                        validation_data=(validation_inputs, validation_labels),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1,
                        callbacks=[early_stopping])  # , tensorboard])

    save_history(path='.', history=history.history)

    # serialize model to JSON
    model_json = model.to_json()
    with open("./model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./model.h5")
    print("Saved model to disk")

    return history, model


def testMe(model, testing_inputs, testing_labels, batch_size):
    """
    Testing function

    Args:
        testing_inputs (numpy.ndarray): Testing set inputs
        testing_labels (numpy.ndarray): Testing set labels
        batch_size (int): Batch size

    Returns: None

    """
    # Evaluate inputs
    test_performance = model.evaluate(testing_inputs, testing_labels, batch_size=batch_size, verbose=1)
    print('test_performance', test_performance)

    y_pred = model.predict(testing_inputs, batch_size=batch_size, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)


seed_everything()
train_df = pd.read_csv("./dataset/train.csv")
train_df = shuffle(train_df)
test_df = pd.read_csv("./dataset/test.csv")
valid_df = pd.read_csv("./dataset/validate.csv")[:20000]
print("Train shape : ", train_df.shape)
print("Validate shape : ", valid_df.shape)
print("Test shape : ", test_df.shape)

# clean the sentences
train_df['cleaned_text'] = train_df['path'].progress_apply(lambda x: clean_sentence(x))
valid_df['cleaned_text'] = valid_df['path'].progress_apply(lambda x: clean_sentence(x))
test_df['cleaned_text'] = test_df['path'].progress_apply(lambda x: clean_sentence(x))

## fill up the missing values
x_train = train_df["cleaned_text"].fillna("_##_").values
x_test = test_df["cleaned_text"].fillna("_##_").values
x_valid = valid_df["cleaned_text"].fillna("_##_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_train))
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_valid = tokenizer.texts_to_sequences(x_valid)

word_index = tokenizer.word_index
## Pad the sentences
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
x_valid = pad_sequences(x_valid, maxlen=maxlen)

## Get the target values
y_train = train_df.cat.values
y_test = test_df.cat.values
y_validate = valid_df.cat.values

# shuffling the data

np.random.seed(SEED)
trn_idx = np.random.permutation(len(x_train))
x_train = x_train[trn_idx]
y_train = y_train[trn_idx]

# missing entries in the embedding are set using np.random.normal so we have to seed here too
seed_everything()
glove_embeddings = load_glove(word_index)
embedding_matrix = glove_embeddings

model = model_cnn(embedding_matrix)

hist, model = trainMe(model=model,
                      training_inputs=x_train,
                      training_labels=y_train,
                      validation_inputs=x_valid,
                      validation_labels=y_validate,
                      epochs=100,
                      batch_size=64,
                      checkpoint_every=100)

testMe(model=model, testing_inputs=x_test, testing_labels=y_test, batch_size=128)
