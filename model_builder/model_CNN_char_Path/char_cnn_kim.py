import numpy as np
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import AlphaDropout
from keras.layers import Convolution1D
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
from keras.models import model_from_json
from keras_radam import RAdam

from data_utils_general import f1_m, recall_m, precision_m, print_result


class CharCNNKim(object):
    """
    Class to implement the Character Level Convolutional Neural Network
    as described in Kim et al., 2015 (https://arxiv.org/abs/1508.06615)

    Their model has been adapted to perform text classification instead of language modelling
    by replacing subsequent recurrent layers with dense layer(s) to perform softmax over classes.
    """

    def __init__(self, input_size, alphabet_size, embedding_size,
                 conv_layers, fully_connected_layers,
                 num_of_classes, dropout_p,
                 optimizer='adam', loss='categorical_crossentropy'):
        """
        Initialization for the Character Level CNN model.

        Args:
            input_size (int): Size of input features
            alphabet_size (int): Size of alphabets to create embeddings for
            embedding_size (int): Size of embeddings
            conv_layers (list[list[int]]): List of Convolution layers for model
            fully_connected_layers (list[list[int]]): List of Fully Connected layers for model
            num_of_classes (int): Number of classes in data
            dropout_p (float): Dropout Probability
            optimizer (str): Training optimizer
            loss (str): Loss function
        """
        self.input_size = input_size
        self.alphabet_size = alphabet_size
        self.embedding_size = embedding_size
        self.conv_layers = conv_layers
        self.fully_connected_layers = fully_connected_layers
        self.num_of_classes = num_of_classes
        self.dropout_p = dropout_p
        self.optimizer = optimizer
        self.loss = loss
        self.LR = 1e-4
        self._build_model()  # builds self.model variable

    def _build_model(self):
        """
        Build and compile the Character Level CNN model

        Returns: None

        """
        # Input layer
        inputs = Input(shape=(self.input_size,), name='sent_input', dtype='int64')
        # Embedding layers
        x = Embedding(self.alphabet_size + 1, self.embedding_size, input_length=self.input_size)(inputs)
        # Convolution layers
        convolution_output = []
        for num_filters, filter_width in self.conv_layers:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=filter_width,
                                 activation='tanh',
                                 name='Conv1D_{}_{}'.format(num_filters, filter_width))(x)
            pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
            convolution_output.append(pool)
        x = Concatenate()(convolution_output)
        # Fully connected layers
        for fl in self.fully_connected_layers:
            x = Dense(fl, activation='relu', kernel_initializer='lecun_normal')(x)
            x = AlphaDropout(self.dropout_p)(x)
        # Output layer
        predictions = Dense(self.num_of_classes, activation='softmax')(x)
        # Build and compile model
        model = Model(inputs=inputs, outputs=predictions)
        # model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[f1_m, recall_m, precision_m])
        model.compile(optimizer=RAdam(learning_rate=self.LR), loss=self.loss, metrics=[f1_m, recall_m, precision_m])
        self.model = model
        print("CharCNNKim model built: ")
        self.model.summary()

    def train(self, training_inputs, training_labels,
              validation_inputs, validation_labels,
              epochs, batch_size, checkpoint_every=100):
        """
        Training function

        Args:
            training_inputs (numpy.ndarray): Training set inputs
            training_labels (numpy.ndarray): Training set labels
            validation_inputs (numpy.ndarray): Validation set inputs
            validation_labels (numpy.ndarray): Validation set labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            checkpoint_every (int): Interval for logging to Tensorboard

        Returns: None

        """
        # Create callbacks
        tensorboard = TensorBoard(log_dir='./log_Kim', histogram_freq=checkpoint_every, batch_size=batch_size,
                                  write_graph=False, write_grads=True, write_images=False,
                                  embeddings_freq=checkpoint_every,
                                  embeddings_layer_names=None
                                  )

        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=5,
                                       verbose=1,
                                       mode='min')

        # Start training
        print("Training CharCNNKim model: ")
        history = self.model.fit(training_inputs,
                                 training_labels,
                                 validation_data=(validation_inputs, validation_labels),
                                 # validation_split=0.2,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 verbose=2,
                                 callbacks=[early_stopping])

        # save_history(path='model_Kim/', history=history.history)

        # # serialize model to JSON
        # model_json = self.model.to_json()
        # with open("model_Kim/model.json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # self.model.save_weights("model_Kim/model.h5")
        # print("Saved model to disk")

        return history

    def test(self,
             testing_inputs,
             testing_labels):
        """
        Testing function

        Args:
            testing_inputs (numpy.ndarray): Testing set inputs
            testing_labels (numpy.ndarray): Testing set labels
            batch_size (int): Batch size

        Returns: None

        """
        # Evaluate inputs
        test_pred_y = np.argmax(self.model.predict(testing_inputs, verbose=True), axis=1)
        test_y = np.argmax(testing_labels, axis=1)

        print('Testing')
        print_result(test_pred_y, test_y)

    def predict(self, input_indices):

        # load json and create model
        json_file = open('model_Zhang/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model_Zhang/model.h5")
        print("Loaded model from disk")

        input_test = np.asarray(input_indices, dtype='int64')
        prediction_probs = self.model.predict(input_test, batch_size=10000, verbose=0)

        # if np.argmax(prediction_probs) == 1:
        #     print('CSA')
        # else:
        #     print('NORMAL')
        # print(prediction_probs)
