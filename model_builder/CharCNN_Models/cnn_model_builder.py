import json

from CharCNN_Models.char_cnn_kim import CharCNNKim
from CharCNN_Models.char_cnn_tcn import CharTCN
from CharCNN_Models.char_cnn_zhang import CharCNNZhang
from CharCNN_Models.data_utils import Data

MIN_DATASET_SIZE = 8000
model = 'tcn'

if __name__ == "__main__":
    # Load configurations
    config = json.load(open("./config.json"))

    # Load training data
    training_data = Data(data_source=config["data"]["training_data_source"],
                         alphabet=config["data"]["alphabet"],
                         input_size=config["data"]["input_size"],
                         num_of_classes=config["data"]["num_of_classes"])
    training_data.load_data(MIN_DATASET_SIZE, train=True)
    training_inputs, training_labels = training_data.get_all_data()

    # Load validation data
    validation_data = Data(data_source=config["data"]["validation_data_source"],
                           alphabet=config["data"]["alphabet"],
                           input_size=config["data"]["input_size"],
                           num_of_classes=config["data"]["num_of_classes"])
    validation_data.load_data(train=True)
    validation_inputs, validation_labels = validation_data.get_all_data()

    # Load Testin data
    testing_data = Data(data_source=config["data"]["testing_data_source"],
                        alphabet=config["data"]["alphabet"],
                        input_size=config["data"]["input_size"],
                        num_of_classes=config["data"]["num_of_classes"])
    testing_data.load_data(train=False)
    testing_inputs_1, testing_labels_1 = testing_data.get_all_data()

    testing_data_2 = Data(data_source=config["data"]["validation_data_source"],
                          alphabet=config["data"]["alphabet"],
                          input_size=config["data"]["input_size"],
                          num_of_classes=config["data"]["num_of_classes"])
    testing_data_2.load_data(train=False)
    testing_inputs_2, testing_labels_2 = testing_data_2.get_all_data()

    # Load model configurations and build model
    if model == "kim":
        model = CharCNNKim(input_size=config["data"]["input_size"],
                           alphabet_size=config["data"]["alphabet_size"],
                           embedding_size=config["char_cnn_kim"]["embedding_size"],
                           conv_layers=config["char_cnn_kim"]["conv_layers"],
                           fully_connected_layers=config["char_cnn_kim"]["fully_connected_layers"],
                           num_of_classes=config["data"]["num_of_classes"],
                           dropout_p=config["char_cnn_kim"]["dropout_p"],
                           optimizer=config["char_cnn_kim"]["optimizer"],
                           loss=config["char_cnn_kim"]["loss"])
    elif model == 'tcn':
        model = CharTCN(input_size=config["data"]["input_size"],
                        alphabet_size=config["data"]["alphabet_size"],
                        embedding_size=config["char_tcn"]["embedding_size"],
                        conv_layers=config["char_tcn"]["conv_layers"],
                        fully_connected_layers=config["char_tcn"]["fully_connected_layers"],
                        num_of_classes=config["data"]["num_of_classes"],
                        dropout_p=config["char_tcn"]["dropout_p"],
                        optimizer=config["char_tcn"]["optimizer"],
                        loss=config["char_tcn"]["loss"])
    elif model == 'zhang':
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
                validation_inputs=validation_inputs,
                validation_labels=validation_labels,
                epochs=config["training"]["epochs"],
                batch_size=config["training"]["batch_size"],
                checkpoint_every=config["training"]["checkpoint_every"])
    model.test(testing_inputs_1, testing_labels_1, testing_inputs_2, testing_labels_2)
