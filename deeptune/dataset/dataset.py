import h5py


def get_char_probabilities(
        dataset: h5py.File,
        dict_encode: dict,
        song_keys: list
):
    probabilities = [0] * len(dict_encode)
    print('Starting char probabilities analysis from song keys...')
    for i, song_key in enumerate(song_keys):
        print(
            'Execution: {:.2f}\r'.format(i / len(song_keys) * 100),
            end=''
        )
        song = list(dataset[song_key][0])
        for char in song:
            probabilities[dict_encode[char]] += 1
    N = sum(probabilities)
    for i in range(len(probabilities)):
        probabilities[i] /= N
    return probabilities


def get_song_key_respecting_dic(
        dataset: h5py.File,
        dictionary: list
) -> list:
    """ Verify which songs in dictionnary respect a given dictionary

    # Arguments:
        dataset (h5py.File): the dataset containing a key->songs mapping
        dictionary (list): the dictionary to use
    # Returns:
        a list of the song keys that respect the dictionary
    """
    song_key_respecting_dic = list()
    print('Starting song analysis from dictionary...')
    for i, song_key in enumerate(list(dataset.keys())[:100]):
        print(
            'Execution: {:.2f}\r'.format(i / len(dataset.keys()) * 100),
            end=''
        )
        song = dataset[song_key][0]
        song_respect_dic = True
        for char in song:
            if char not in dictionary:
                song_respect_dic = False
                break
        if song_respect_dic:
            song_key_respecting_dic.append(song_key)
    return song_key_respecting_dic
