import h5py
import keras
import numpy as np
import os

from .generators import DataGenerator
from .generators import encode, decode

from .losses import padded_categorical_crossentropy, padded_categorical_hinge

from keras.layers import Input, Add, LSTM, Dense
from keras.layers.core import Activation
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.callbacks import CSVLogger, ModelCheckpoint


class LyricaModel():
    """ A TCN Lyrica generator model

    # Arguments:
        dic_size (int): dictionnary size
        length (int): number of stacked conv1d layer
        filters (int): number of filters for each layer
        kernel_reg (float): l2 conv1d kernel regularizer
    """

    def __init__(
            self,
            dic_size: int,
            length: int,
            filters: int,
            kernel_size: int,
            kernel_reg: float,
            pretrained_weight_path: str = None
    ):
        self.input_shape = (None, dic_size)
        self.length = length
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel_reg = kernel_reg

        self.model = self._build_TCN_model()

        optimizer = Adam(
            lr=0.001
        )

        self.model.compile(
            loss=padded_categorical_hinge,
            optimizer=optimizer,
            metrics=['accuracy']
        )

        self.model.summary(line_length=120)
        if pretrained_weight_path is not None:
            self.model.load_weights(pretrained_weight_path)

    def _build_TCN_model(
            self
    ) -> Model:
        """ build the TCN model

        # Returns:
            a keras Model object
        """
        previous_layer = Input(
            self.input_shape,
            name='input_layer'
        )
        input_layer = previous_layer
        for i in range(self.length):
            if i == self.length - 1:
                filters = self.input_shape[-1]
                kernel_size = 9
                activation = 'softmax'
            else:
                filters = self.filters
                kernel_size = self.kernel_size
                activation = 'relu'
            previous_layer = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='causal',
                dilation_rate=2 ** i,
                use_bias=True,
                kernel_regularizer=l2(self.kernel_reg),
                kernel_initializer='glorot_normal',
                name='conv1d_{}'.format(i + 1)
            )(previous_layer)
            previous_layer = Activation(
                activation,
                name='activation_{}'.format(i + 1)
            )(previous_layer)

        model = Model(
            inputs=input_layer,
            outputs=previous_layer
        )
        return model

    def train_and_valid(
            self,
            hdf5_dataset: h5py.File,
            train_keys_list: list,
            valid_keys_list: list,
            batch_size: int,
            nb_epochs: int,
            dic_encode: dict,
            save_folder: str = None
    ):
        """ Train the TCN model

        Will save the weights file as weights.h5 from the best model in terms of the
        training loss

        # Arguments:
            hdf5_dataset (h5py.File): dataset that contains the songs
            train_keys_list (list): list of strings of the hdf5_dataset keys for train
            valid_keys_list (list): list of strings of the hdf5_dataset keys for valid
            batch_size (int): batch size
            nb_epochs (int): number of epochs to train
            dic_encode (dict): dictionnary mapping characters to integer
        """
        training_generator = DataGenerator(
            hdf5_dataset=hdf5_dataset,
            song_keys_list=train_keys_list,
            batch_size=batch_size,
            dic_encode=dic_encode,
            shuffle=True
        )
        validation_generator = DataGenerator(
            hdf5_dataset=hdf5_dataset,
            song_keys_list=valid_keys_list,
            batch_size=batch_size,
            dic_encode=dic_encode,
            shuffle=False
        )

        if save_folder is None:
            save_folder = 'Lvl{}_Nf{}_Ks{}'.format(
                self.length,
                self.filters,
                self.kernel_size
            )
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        path_csv = os.path.join(save_folder, 'training_logs.csv')
        path_weights = os.path.join(save_folder, 'weights.h5')
        callback_csv_logger = CSVLogger(path_csv, append=False)
        callback_model_save = ModelCheckpoint(
            path_weights,
            monitor='loss',
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1
        )

        callbacks = [
            callback_csv_logger,
            callback_model_save
        ]

        self.model.fit_generator(
            generator=training_generator,
            steps_per_epoch=np.ceil(len(train_keys_list) / batch_size / 10),
            epochs=nb_epochs,
            validation_data=validation_generator,
            validation_steps=np.ceil(len(valid_keys_list) / batch_size),
            callbacks=callbacks
        )

    def predict(
            self,
            song: str,
            chars_to_predict: int,
            list_decode: list,
            dic_encode: dict
    ):
        """ Predict the next characters of a song

        # Arguments:
            song (str): the begginning of the song
            chars_to_predict (int): the number of characters to predict
            list_decode (list): a list mapping index to characters
            dic_encode (dict): a dict mapping characters to integers
        """
        song_one_hot = encode(song, dic_encode)
        for _ in range(chars_to_predict):
            next_chr = self.model.predict(song_one_hot[np.newaxis])[0]
            song_one_hot = np.concatenate(
                (
                    song_one_hot,
                    next_chr[np.newaxis]
                )
            )
        song = decode(song_one_hot, list_decode)
        return song
