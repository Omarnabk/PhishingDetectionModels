import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from model_builder.CharCNN_Models.char_cnn_kim import CharCNNKim
from model_builder.CharCNN_Models.char_cnn_tcn import CharTCN
from model_builder.CharCNN_Models.char_cnn_zhang import CharCNNZhang
from model_builder.CharCNN_Models.data_utils import Data
from sklearn.model_selection import train_test_split

DATASET_ROOT_PATH = '../model_builder/dataset/'


def write2csv(file_name, all_x, all_y):
    with open(file_name, 'w', encoding='utf-8') as wrt:
        for x, y in zip(all_x, all_y):
            wrt.write('{}\t{}\n'.format(x, y))


def load_json(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
        return data


tf.flags.DEFINE_string("model", "zhang", "Specifies which model to use: char_cnn_zhang or char_cnn_kim")
FLAGS = tf.flags.FLAGS

if __name__ == "__main__":
    # Load configurations
    config = json.load(open("../model_builder/CharCNN_Models/config.json"))

    model_architecture_name = FLAGS.model

    x_class_ph = load_json(DATASET_ROOT_PATH + 'Phishing-20K.json')
    y_class_ph = len(x_class_ph) * [0]

    x_class_lg = load_json(DATASET_ROOT_PATH + 'Legitimate-20K.json')
    y_x_class_lg = len(x_class_lg) * [1]

    x_class_lg_login = load_json(DATASET_ROOT_PATH + 'LegitimateLogin-20K.json')
    y_x_class_lg_login = len(x_class_lg_login) * [2]

    X = np.array(x_class_ph + x_class_lg)
    y = np.array(y_class_ph + y_x_class_lg)

    X, _, y, _ = train_test_split(X, y, test_size=0.0, random_state=42)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for fold_id, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Load training data
        training_data = Data(alphabet=config["data"]["alphabet"],
                             input_size=config["data"]["input_size"],
                             num_of_classes=len(set(y_train)))
        training_inputs, training_labels = training_data.get_all_data(X_train, y_train)

        # # Load validation data
        # validation_data = Data(data_source=config["data"]["validation_data_source"],
        #                        alphabet=config["data"]["alphabet"],
        #                        input_size=config["data"]["input_size"],
        #                        num_of_classes=config["data"]["num_of_classes"])
        # validation_inputs, validation_labels = validation_data.get_all_data()

        # Load testing data
        testing_data = Data(alphabet=config["data"]["alphabet"],
                            input_size=config["data"]["input_size"],
                            num_of_classes=len(set(y_train)))

        testing_inputs, testing_labels = testing_data.get_all_data(X_test, y_test)

        # Load model configurations and build model
        if FLAGS.model == "kim":
            model = CharCNNKim(input_size=config["data"]["input_size"],
                               alphabet_size=config["data"]["alphabet_size"],
                               embedding_size=config["char_cnn_kim"]["embedding_size"],
                               conv_layers=config["char_cnn_kim"]["conv_layers"],
                               fully_connected_layers=config["char_cnn_kim"]["fully_connected_layers"],
                               num_of_classes=config["data"]["num_of_classes"],
                               dropout_p=config["char_cnn_kim"]["dropout_p"],
                               optimizer=config["char_cnn_kim"]["optimizer"],
                               loss=config["char_cnn_kim"]["loss"])
        elif FLAGS.model == 'tcn':
            model = CharTCN(input_size=config["data"]["input_size"],
                            alphabet_size=config["data"]["alphabet_size"],
                            embedding_size=config["char_tcn"]["embedding_size"],
                            conv_layers=config["char_tcn"]["conv_layers"],
                            fully_connected_layers=config["char_tcn"]["fully_connected_layers"],
                            num_of_classes=config["data"]["num_of_classes"],
                            dropout_p=config["char_tcn"]["dropout_p"],
                            optimizer=config["char_tcn"]["optimizer"],
                            loss=config["char_tcn"]["loss"])
        else:
            model = CharCNNZhang(input_size=config["data"]["input_size"],
                                 alphabet_size=config["data"]["alphabet_size"],
                                 embedding_size=config["char_cnn_zhang"]["embedding_size"],
                                 conv_layers=config["char_cnn_zhang"]["conv_layers"],
                                 fully_connected_layers=config["char_cnn_zhang"]["fully_connected_layers"],
                                 num_of_classes=config["data"]["num_of_classes"],
                                 threshold=config["char_cnn_zhang"]["threshold"],
                                 dropout_p=config["char_cnn_zhang"]["dropout_p"],
                                 optimizer=config["char_cnn_zhang"]["optimizer"],
                                 loss=config["char_cnn_zhang"]["loss"])

        # Train model
        model.train(training_inputs=training_inputs,
                    training_labels=training_labels,
                    validation_inputs=None,
                    validation_labels=None,
                    testing_inputs=testing_inputs,
                    testing_labels=testing_labels,
                    fold_id=fold_id,
                    epochs=config["training"]["epochs"],
                    batch_size=config["training"]["batch_size"],
                    model_name=f'../compiled_models/char_{model_architecture_name}_model/',
                    report_name=f'../compiled_models/char_{model_architecture_name}_model/clf_{fold_id}.csv',

                    )
        model.test(testing_inputs=testing_inputs,
                   testing_labels=testing_labels
                   )
