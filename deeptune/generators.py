import numpy as np
import h5py

from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence


class DataGenerator(Sequence):
    """Generates data for Keras

    # Arguments:
        hdf5_dataset (h5py.File): dataset that contains the songs
        song_keys_list (list): list of str of the hdf5_dataset keys
        batch_size (int): batch size
        dic_encode (dict): dictionnary mapping characters to integer
        shuffle (bool): shuffle the keys list after each epoch
        mask_token (float): token used for masking padded samples for batching
    """
    def __init__(
            self,
            hdf5_dataset: h5py.File,
            song_keys_list: list,
            batch_size: int,
            dic_encode: dict,
            shuffle: bool = True,
            mask_token: float = -1.0
    ):
        self.hdf5_dataset = hdf5_dataset
        self.song_keys_list = song_keys_list
        self.batch_size = batch_size
        self.dic_encode = dic_encode
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(
            self
    ) -> int:
        """ Length of the dataset in batch

        # Return
            length of dataset in batch
        """
        return int(np.floor(len(self.song_keys_list) / self.batch_size))

    def __getitem__(
            self,
            index: int,
    ) -> (np.ndarray, np.ndarray):
        """Generate one batch of data

        # Arguments
            index (int): index of the batch
        # Return
            batch of samples (network inputs, ground truths)
        """
        song_keys = self.song_keys_list[
            index * self.batch_size:(index + 1) * self.batch_size
        ]

        return self._get_batch(song_keys)

    def on_epoch_end(
            self
    ):
        """shuffle the keys list after each epoch
        """
        if self.shuffle:
            np.random.shuffle(self.song_keys_list)

    def _get_batch(
            self,
            song_keys: list
    ) -> (np.ndarray, np.ndarray):
        """ Generate one batch of data from song keys

        # Arguments:
            song_keys (list): list of the song keys
        # Return:
            batch of samples (network inputs, ground truths)
        """
        previous_chars_batch, next_chars_batch = list(), list()
        for song_key in song_keys:
            previous_chars, next_chars = self._get_sample(song_key)
            previous_chars_batch.append(previous_chars)
            next_chars_batch.append(next_chars)
        previous_chars_batch = pad_sequences(
            sequences=previous_chars_batch,
            dtype=np.float32,
            padding='pre',
            value=0.0
        )
        next_chars_batch = pad_sequences(
            sequences=next_chars_batch,
            dtype=np.float32,
            padding='pre',
            value=0.0
        )
        return previous_chars_batch, next_chars_batch

    def _get_sample(
            self,
            song_key: str
    ) -> (np.ndarray, np.ndarray):
        """ get a single song sample from key

        # Arguments:
            song_key (str): the song key
        # Returns:
            the sample (network input, ground truth)
        """
        song = self.hdf5_dataset[song_key][0]
        chars_one_hot = encode(song, self.dic_encode)
        previous_chars = chars_one_hot[:-1]
        next_chars = chars_one_hot[1:]
        return previous_chars, next_chars

    def random_sub_sample(
            self,
            song: np.ndarray,
            min_length: int = 64,
            max_length: int = 256
    ) -> np.ndarray:
        if min_length >= len(song):
            return song
        if max_length > len(song):
            max_length = len(song)
        sub_song_length = np.random.randint(min_length, max_length)
        start_index = np.random.randint(len(song) - sub_song_length + 1)
        return song[start_index: start_index + sub_song_length]


def encode(
        song: str,
        dic_encode: dict
) -> np.ndarray:
    """ Encode a string to one hot

    # Arguments:
        song (str): song to encode
        dic_encode (dict): a dict mapping characters to integers
    # Returns:
        one hot encoded song
    """
    chars = list(song)
    chars_encoded = [dic_encode[char] for char in chars]
    chars_one_hot = np.zeros(
        (len(chars_encoded), len(dic_encode)),
        np.float32
    )
    for index, char_one_hot in zip(chars_encoded, chars_one_hot):
        char_one_hot[index] = 1.0
    return chars_one_hot


def decode(
        song_encoded: np.ndarray,
        list_decode: list
) -> str:
    """ decode a one hot encoded song

    # Arguments:
        song_encoded (np.ndarray): the one hot song
        list_decode (list): a list containing the mapping of index to character
    # Returns:
        the song in string format
    """
    return ''.join([
        list_decode[np.argmax(one_hot_char)] for one_hot_char in song_encoded])
