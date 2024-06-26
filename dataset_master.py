import os
import numpy as np
import pandas as pd
import hd5f_getters as GETTERS
import librosa
import librosa.display
import soundfile
import audioread
import matplotlib.pyplot as plt
from tqdm import tqdm


import utils
from wmf_script import WMF_Model
from utilities.constants import(
    DEFAULT_FMA_METADATA_LOCATION,
    FMA_SONG_LOCATION,
    TRIPLET_DATA_LOCATION,
    MILLION_SONG_CSV_LOCATION,
    SAVED_MODEL_LOCATION,
    DEFAULT_DATA_LOCATION,
    TINY_DATA_LOCATION
)

class DatasetMaster():
    def __init__(self):
        self.df_train_triplets = None
        self.df_million_song = None
        self.matched_tracks = None
        self.master_df = self.create_master()
        
        self.WMF = WMF_Model()
        self.WMF = self.WMF.load(SAVED_MODEL_LOCATION+"/WMF/wmf_on_best_TT.pkl")
        self.CNN_Train_Data = self.create_CNN_data()
        
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

    def find_fma_song(self, fma_track_index):
        # Given the fma_track index return a path to the actual mp3 of the song
        folder = str(fma_track_index // 1000)
        
        filename = str(fma_track_index)
        if fma_track_index < 1000:
            folder = f"000"
            filename = f"000{filename}"
        elif fma_track_index < 10000:
            folder = f"00{folder}"
            filename = f"00{filename}"
        elif fma_track_index < 100000:
            folder = f"0{folder}"
            filename = f"0{filename}"
        filename += ".mp3"
        
        location = f"{FMA_SONG_LOCATION}/{folder}/{filename}"
        return location
    
    def create_mel_spectrogram(self, audio_path, offset=0, duration=None, n_mels=128, hop_length=512, win_length=1024, plot=False):
        '''
        Take in the path to an MP3 and use the librosa library to do an FFT (I think) and
        convert the raw audio into an audio time series.
        The time series goes into a function that computes a mel-scaled spectogram
        The power spectrogram (amplitude squared) then gets converted to decibel (dB) units
        https://pnsn.org/spectrograms/what-is-a-spectrogram is helpful
        
        Returned is the decibel scaled spectrogram for the CNN
        '''
        try:
            y, sr = librosa.load(audio_path, sr=None, offset=offset, duration=duration)  # keeps original sampling rate. offset is when to start loading and dur. is how long
        except (soundfile.LibsndfileError, audioread.NoBackendError) as e:
            # print(f"Bad file at: {audio_path} - Error: {e}")
            return None

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, win_length=win_length)

        S_dB = librosa.power_to_db(S, ref=np.max)

        if plot:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel-frequency spectrogram')
            plt.tight_layout()
            plt.show()

        return S_dB.astype(np.float16)
    
    def create_CNN_data(self):        
        count = 0
        total = len(self.master_df['track_id'])
        train_data = [[], [], [], []] # train_data, train_label, val_data, val, label
        if os.path.exists('train_data') and 0:
            print("LOADING THE DATA FROM TEXT FILES")
            train_data = [np.empty((18503, 128, 259)), np.empty((18503, 50)), np.empty((2230, 128, 259)), np.empty((2230, 50))]
            for _, _, files in os.walk('train_data'):
                for file in files:
                    indices = file[:-4].split("_")
                    train_data[int(indices[0])][int(indices[1])] = np.loadtxt(f"train_data/{file}", dtype=np.float16)
            return train_data

        for song in tqdm(self.master_df.iterrows(), total=self.master_df.shape[0]):
            count += 1
            
            index_into_fma_track_df = song[1]['index_into_fma_track_df']
            song_pieces = []
            for start in range(0, 30, 5):
                train_data_single = self.create_mel_spectrogram(self.find_fma_song(index_into_fma_track_df), offset=start, duration=3)
                
                if train_data_single is None:
                    break
                if train_data_single.shape[1] != 259:
                    continue
                
                song_pieces.append(train_data_single)
                count += 1

            index_into_TT = -1
            
            for index, million_song in enumerate(self.df_million_song.to_numpy()):
                if million_song[0] == song[1]['track_id'] and million_song[1] == song[1]['track_title']:
                    index_into_TT = index
            if index_into_TT == -1:
                raise Exception
            
            # Add to the validation data roughly 10% of all song samples
            size = len(song_pieces)
            val_index = -1
            if size > 3:
                val_index = np.random.randint(size)
            
            for index, song_data in enumerate(song_pieces):
                if index == val_index:
                    train_data[2].append(np.array(song_data))
                    train_data[3].append(self.WMF.get_item_vectors()[index_into_TT])
                else:
                    train_data[0].append(np.array(song_data))
                    train_data[1].append(self.WMF.get_item_vectors()[index_into_TT])
        
        print(f"From {total} songs {count} data samples were created")
        train_data[0] = np.array(train_data[0])
        train_data[1] = np.array(train_data[1])
        train_data[2] = np.array(train_data[2])
        train_data[3] = np.array(train_data[3])
        return train_data

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
        # Create a intersection of Million Song Dataset songs and songs in the FMA audio samples
        # df_all_triplet_songs = pd.read_csv(TRIPLET_DATA_LOCATION, sep='\t', header=None, names=['userID', 'fma_track_id', 'rating'])
        intersection_tt_MSD_songs = []
        
        self.create_million_song()
        print(self.df_million_song.columns)
        million_song_list = list(self.df_million_song['track_id'])

        for song in tqdm(million_song_list):
            if song in million_song_list:
                intersection_tt_MSD_songs.append(song)
        
        return intersection_tt_MSD_songs

