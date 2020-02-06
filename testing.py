"""
Midi message:
    type (str): {'note_on', 'note_off'}
    channel (int): [0-12]
    note (int): [0-127]
    velocity (int): [0-127]
    time (int): ticks from last message
"""
import matplotlib.pyplot as plt
import mido
import numpy as np
import os
import pandas as pd
import deeptune

dataset_path = 'deeptune/dataset/data/vgmusic/console/nintendo/snes'
output_file_path = dataset_path + '/ticks_per_beat.txt'


def make_ticks_per_beat_list(
        dataset_path: str,
        output_file_path: str
):
    with open(output_file_path, 'w') as f:
        for root, directory, filesnames in os.walk(dataset_path):
            for filename in filesnames:
                file_path = root + '/' + filename
                try:
                    midi_file = mido.MidiFile(file_path)
                    f.write(file_path + ', {}\n'.format(midi_file.ticks_per_beat))
                except Exception as e:
                    print(e, ' in file ' + file_path)


def create_sample_from_midi(
        midi_path: str,
):
    midi_file = mido.MidiFile(midi_path)
    for message in midi_file.play:
        print(message)
    # for track in midi_file.tracks:
    #     for msg in track:
    #         print(msg)


def count_ticks_per_beat_from_csv(
        file_path: str,
        top_n: int = 5
):
    df = pd.read_csv(output_file_path, header=None)
    ticks_per_beat = np.array(df[1])
    unique_elements, counts_elements = np.unique(
        ticks_per_beat,
        return_counts=True
    )
    unique_sorted = unique_elements[np.argsort(counts_elements)[::-1][:top_n]]
    counts_sorted = np.sort(counts_elements)[::-1][:top_n]
    print('Max number of song with same ticks per beat:')
    print('\tTicks per beat: ', unique_sorted)
    print('\tNo elements: ', counts_sorted)

# make_ticks_per_beat_list(
#     dataset_path,
#     output_file_path
# )

# count_ticks_per_beat_from_csv(
#     output_file_path
# )


scrapper = deeptune.dataset.scraper.metallyrica(
    'deeptune/dataset/data/metallyrica/metallyrica.hdf5'
)
scrapper.scrap(min_song_length=50)