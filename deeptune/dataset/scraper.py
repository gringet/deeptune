from bs4 import BeautifulSoup
from os import makedirs, remove
from os.path import isdir, join, isfile

import numpy as np
import h5py
import urllib
import requests


class vgmusic():

    def __init__(
            self,
            dataset_path: str
    ):
        self.dataset_path = dataset_path

    def scrap(
            self
    ):
        url = 'https://www.vgmusic.com/'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.findAll('a')

        for link in links:
            link = str(link.get('href'))
            if '/music/' in link:
                sub_url = url + link[2:]
                sub_dataset_path = self.dataset_path + link[7:]
                response = requests.get(sub_url)
                sub_soup = BeautifulSoup(response.text, 'html.parser')
                sub_links = sub_soup.findAll('a')
                for sub_link in sub_links:
                    sub_link = str(sub_link.get('href'))
                    if sub_link.endswith('.mid'):
                        download_url = sub_url + sub_link
                        if not isdir(sub_dataset_path + sub_link[0]):
                            makedirs(sub_dataset_path + sub_link[0])
                        filename = sub_dataset_path + sub_link[0] + '/' + sub_link
                        if not isfile(filename):
                            urllib.request.urlretrieve(
                                url=download_url,
                                filename=filename
                            )


class metallyrica():

    def __init__(
            self,
            dataset_path: str
    ):
        self.dataset_path = dataset_path

    def scrap(
            self,
            min_song_length: int = 50,
    ):
        if isfile(self.dataset_path):
            remove(self.dataset_path)
        dset = h5py.File(self.dataset_path, 'w')
        song_id = 0
        url = 'http://www.metallyrica.com/'
        # band are organized by first char in name
        first_chrs = [chr(i) for i in range(97, 97+26)]
        first_chrs.append('0')
        for first_chr in first_chrs:
            print(first_chr)
            first_chr_url = url + first_chr + '.html'
            response = requests.get(first_chr_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            band_links = soup.findAll('a')
            for band_link in band_links:
                band_link = str(band_link.get('href'))
                filter_band_link = first_chr + '/' + first_chr
                # verify if it's really a band link
                if band_link.startswith(filter_band_link):
                    band_url = url + band_link
                    response = requests.get(band_url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    song_links = soup.findAll('a')
                    for song_link in song_links:
                        song_link = str(song_link.get('href'))
                        # verify if it's really a song link
                        if song_link.startswith('../lyrica'):
                            # The song links goes to the album link, scrap it only once.
                            if song_link.endswith('#1'):
                                album_url = url + song_link[3:-2]
                                songs = self._scrap_album(album_url)
                                if songs is not None:
                                    for song in songs:
                                        if len(song) > min_song_length:
                                            dset.create_dataset(
                                                name='{:09}'.format(song_id),
                                                shape=(1,),
                                                dtype=h5py.string_dtype(),
                                                data=song
                                            )
                                            song_id += 1
        dset.close()

    def _scrap_album(
            self,
            album_url: str
    ) -> list:
        response = requests.get(album_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        lines = soup.find_all(text=True)
        lines = self._filter_album_text(lines)
        if lines is None:
            return None
        songs_text, song_text = list(), str()
        song_start_tokens = [str(i) + '. \n' for i in range(100)]
        song_start_tokens.append('\xa0\n')
        skip_next_line = False
        song_id = 0
        for line in lines:
            if skip_next_line:
                skip_next_line = False
                continue
            elif line in song_start_tokens:
                if not song_id == 0:
                    songs_text.append(song_text)
                    song_text = str()
                song_id += 1
                skip_next_line = True
            else:
                song_text += line
        return songs_text

    def _filter_album_text(
            self,
            lines: list
    ) -> list:
        start_index, end_index = 0, 0
        for i, line in enumerate(lines):
            if line.startswith('google_ad_client'):
                start_index = i + 1
            elif line.startswith('<!--\n\tgoogle_ad_client'):
                end_index = i
        if (start_index >= end_index):
            return None
        # supress non songs related lines
        lines = lines[start_index:end_index]
        # supress empty lines at the end of the song
        lines = [line for line in lines if not line == '\n']
        # replace empty lines between paragraph
        lines = [line if not line == ' ' else '\n' for line in lines]
        # add end lines characters
        lines = [line + '\n' if not line == '\n' else line for line in lines]
        # supress back lines characters
        lines = [line.replace('\r', '') for line in lines]
        return lines
