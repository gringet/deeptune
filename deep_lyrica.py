import deeptune
import h5py
import numpy as np
import random

dataset_path = 'deeptune/dataset/data/metallyrica/metallyrica_complete.hdf5'
song_keys_path = 'deeptune/dataset/data/metallyrica/song_keys.txt'
dictionary = [chr(i) for i in range(97, 97 + 26)]
dictionary += [chr(i) for i in range(65, 65 + 26)]
dictionary += [',', ' ', '.', '?', '!', "'", '\n']
dataset = h5py.File(dataset_path, 'r')

# Get song keys that respect given dictionary
# songs = deeptune.dataset.dataset.get_song_key_respecting_dic(
#     dataset,
#     dictionary
# )

# Write the dictionary and song keys to a file
# with open(song_keys_path, 'w') as f:
#     for char in dictionary:
#         f.write(char)
#     for key in songs:
#         f.write(key)
#         f.write('\n')

# Use a given file with the dictionary and the song keys to train a model
with open(song_keys_path, 'r') as f:
    list_decode = list(f.readline())
    dic_encode = dict(zip(list_decode, list(range(len(list_decode)))))
    song_keys = [line[:-1] for line in f.readlines()]

    for length in [7]:
        for kernel_size in [3]:
            for filters in [128]:
                for cross_valid in range(5):
                    if cross_valid > 0:
                        random.seed(1337 + cross_valid)
                        random.shuffle(song_keys)
                    save_folder = 'Lvl{}_Nf{}_Ks{}_{}'.format(
                        length,
                        filters,
                        kernel_size,
                        cross_valid
                    )
                    model = deeptune.model.LyricaModel(
                        dic_size=len(list_decode),
                        length=length,
                        filters=filters,
                        kernel_size=kernel_size,
                        kernel_reg=0.0
                    )
                    model.train_and_valid(
                        hdf5_dataset=dataset,
                        train_keys_list=song_keys[:int(len(song_keys) * 0.8)],
                        valid_keys_list=song_keys[int(len(song_keys) * 0.8):],
                        batch_size=32,
                        nb_epochs=10,
                        dic_encode=dic_encode,
                        save_folder=save_folder
                    )
