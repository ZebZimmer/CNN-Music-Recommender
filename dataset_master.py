import os
import numpy as np
import pandas as pd
import hd5f_getters as GETTERS
from tqdm import tqdm

import utils
from utilities.constants import(
    DEFAULT_FMA_METADATA_LOCATION,
    FMA_SONG_LOCATION,
    TRIPLET_DATA_LOCATION,
    MILLION_SONG_CSV_LOCATION,
    DEFAULT_DATA_LOCATION,
    TINY_DATA_LOCATION
)

class DatasetMaster():
    def __init__(self):
        self.df_train_triplets = None
        self.df_million_song = None
        self.matched_tracks = None
        self.master_df = self.create_master()
        
    def create_master(self):
        self.create_train_triplets()
        self.create_million_song()
        self.refine_df_million_song()
        
        self.create_matched_of_million_song_and_fma_tracks()
        
        master_df = pd.DataFrame(columns=['track_id', 'track_title', 'artist_name_x', 'play_count', 'index_into_fma_track_df', 'artist_name_y'])
        fma_audio_samples_list = self.get_filenames()
        for index, row in self.matched_tracks.iterrows():
            if row['index_into_fma_track_df'] in fma_audio_samples_list:
                master_df = pd.concat([master_df, pd.DataFrame([row])])
        
        print(f"Found: {master_df.shape[0]} songs between the free music archive (audio samples), train_triplets and the million song dataset")
        
        return master_df
        
    def create_train_triplets(self):
        # read in the data triples
        self.df_train_triplets = pd.read_csv(TRIPLET_DATA_LOCATION, sep='\t', header=None, names=['userID', 'itemID', 'rating'])
        
    def refine_df_million_song(self):
        # we're going to find all the song id's in df_million_song that are also in the triplets
        unique_triplet_item_ids = set(self.df_train_triplets["itemID"])
        unique_million_song_subset_ids = set(self.df_million_song["track_id"])
        items_in_both = unique_triplet_item_ids & unique_million_song_subset_ids
        print(f'Found: {len(items_in_both)} songs between the train_triplets and the million song dataset') 
        
        self.df_million_song = self.df_million_song[self.df_million_song['track_id'].isin(items_in_both)]
        
    def create_million_song(self):
        # process the files then check out the dataframe
        self.df_million_song = pd.read_csv(MILLION_SONG_CSV_LOCATION)
        
    def create_matched_of_million_song_and_fma_tracks(self):
        # Load metadata and features.
        # these objects are all pandas dataframes
        tracks = utils.load(f'{DEFAULT_FMA_METADATA_LOCATION}/tracks.csv')

        resetted_tracks = tracks.reset_index()
        resetted_tracks.rename(columns={'track_id': 'index_into_fma_track_df'}, inplace=True)
        
        # Now normalize both dfs: process the strings and find matches between the DFs
        # extract our desired fma columns

        fma_tracks = resetted_tracks[[('index_into_fma_track_df', ''), ('track', 'title'), ('artist', 'name')]].copy()
        fma_tracks.columns = ['index_into_fma_track_df', 'track_title', 'artist_name']

        # normalize the fma data: lowercase and remove special chars
        fma_tracks['track_title'] = fma_tracks['track_title'].str.lower().str.replace('[^\w\s]', '', regex=True)
        fma_tracks['artist_name'] = fma_tracks['artist_name'].str.lower().str.replace('[^\w\s]', '', regex=True)

        # normalize text data for Million Song subset the same way
        self.df_million_song['track_title'] = self.df_million_song['track_title'].str.lower().str.replace('[^\w\s]', '', regex=True)
        self.df_million_song['artist_name'] = self.df_million_song['artist_name'].str.lower().str.replace('[^\w\s]', '', regex=True)

        # merge DataFrames on 'track_title' and 'artist_name'
        self.matched_tracks = pd.merge(self.df_million_song, fma_tracks, on=['track_title'], how='inner')
        
        print(f"Found: {self.matched_tracks.shape[0]} songs between the free music archive (full track list), train_triplets and the million song dataset")
    
    
    def get_filenames(self):
        # Get every track that is in the FMA Song location
        fma_tracks = []
        for _, dirs, files in os.walk(FMA_SONG_LOCATION):
            for file in files:
                fma_tracks.append(int(file[:-4]))
        return fma_tracks
    
    def find_fma_song(fma_track_index):
        # Given the fma_track index return a path to the actual mp3 of the song
        folder = str(fma_track_index // 1000)
        
        filename = str(fma_track_index)
        if fma_track_index < 100000:
            folder = f"0{folder}"
            filename = f"0{filename}"
        filename += ".mp3"
        
        location = f"{FMA_SONG_LOCATION}/{folder}/{filename}"
        return location

    
    def test_one_file(self, filepath: str) -> None:
        """
        Playing around to see what we can pull out of one file.
        Example usage: 
            >>> filepath = './millionsongsubset/A/A/A/TRAAAAW128F429D538.h5'
            >>> test_one_file(filepath)
        """
        h5 = GETTERS.open_h5_file_read(filepath)
        num_songs = GETTERS.get_num_songs(h5)
        track_id = GETTERS.get_track_id(h5)
        song_id = GETTERS.get_song_id(h5)
        track_title = GETTERS.get_title(h5)
        play_count = GETTERS.get_song_hotttnesss(h5)
        artist_name = GETTERS.get_artist_name(h5)

        print(f'BEFORE decode(): {track_id = }, {track_title = }, {artist_name = }, {play_count = }, {song_id = }')
        print(f'AFTER decode(): {track_id.decode() = }, {track_title.decode() = }, {artist_name.decode() = }, {play_count = }, {song_id.decode() = }')
        print(f'{num_songs = } for {filepath = }')

    def process_files(self, base_dir: str) -> pd.DataFrame:
        """
        Process all the millionsongsubset files and return a dataframe
            with columns: ['track_id', 'track_title', 'artist_name', 'play_count'].
        This will take a few minutes.
        """

        play_data = []
        print('Processing files...')
        for root, dirs, files in os.walk(base_dir):
            # Loop over all .h5 files
            h5_files = [f for f in files if f.endswith('.h5')]
            for file in h5_files:
                file_path = os.path.join(root, file)
                h5 = GETTERS.open_h5_file_read(file_path)
                try:
                    # loop over all the songs in this one h5 file
                    num_songs = GETTERS.get_num_songs(h5)
                    for song_idx in range(num_songs):
                        # get relevant information from the dataset
                        # track_id = GETTERS.get_track_id(h5, songidx=song_idx).decode()
                        song_id = GETTERS.get_song_id(h5).decode()
                        track_title = GETTERS.get_title(h5, songidx=song_idx).decode()
                        artist_name = GETTERS.get_artist_name(h5, songidx=song_idx).decode()

                        # using song_hotttnesss as a proxy for play counts
                        play_count = GETTERS.get_song_hotttnesss(h5, songidx=song_idx)
                        if np.isnan(play_count):
                            play_count = 0
                    play_data.append([song_id, track_title, artist_name, play_count])

                finally:
                    h5.close()

        return pd.DataFrame(play_data, columns=['track_id', 'track_title', 'artist_name', 'play_count'])
    
    def get_WMF_compatible_tracks(self):
        # Create a intersection of train triple songs and songs in the FMA audio samples
        # df_all_triplet_songs = pd.read_csv(TRIPLET_DATA_LOCATION, sep='\t', header=None, names=['userID', 'fma_track_id', 'rating'])
        intersection_tt_MSD_songs = []
        
        self.create_million_song()
        print(self.df_million_song.columns)
        million_song_list = list(self.df_million_song['track_id'])
  
        for song in tqdm(million_song_list):
            if song in million_song_list:
                intersection_tt_MSD_songs.append(song)
        
        return intersection_tt_MSD_songs

