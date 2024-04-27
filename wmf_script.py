# Using Cornac and ideas from https://github.com/ddoeunn/Weighted-Regularized-Matrix-Factorization/blob/main/WRMF/wrmf.py
# WMF github link for reference: https://github.com/PreferredAI/cornac/blob/master/cornac/models/wmf/recom_wmf.py

import pandas as pd
import cornac
from cornac.models import WMF

from utilities.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    TRIPLET_DATA_LOCATION,
)

class WMF_Model(WMF):
    def __init__(self, k=50, learning_rate=0.001, max_iter=100, verbose=True):
        super().__init__(k=k, learning_rate=learning_rate, max_iter=max_iter, verbose=verbose)
        self.train_set = None

    def prepare_cornac_data(self, data):
        return cornac.data.Dataset.from_uir(
            data[[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL]].itertuples(index=False)
        )

    def train_cornac(self, data=TRIPLET_DATA_LOCATION):
        '''
        Get the raw dataset (train_triplets.txt) from the default location. Then parse it into a pandas dataframe.
        Pass the dataframe to cornac's Dataset builder function .fromuir()
        Train the model on that Dataset.
        '''
        df_train_triplets = pd.read_csv(data, sep='\t', header=None, names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL])
        print(df_train_triplets.head())
        self.train_set = self.prepare_cornac_data(df_train_triplets)
        print("Done preparing dataset, now training")
        self.fit(self.train_set)