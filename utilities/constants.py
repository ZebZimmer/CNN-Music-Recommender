# Sourced from: https://github.com/ddoeunn/Weighted-Regularized-Matrix-Factorization/blob/main/utils/common/constants.py

DEFAULT_USER_COL = "userID"
DEFAULT_ITEM_COL = "itemID"
DEFAULT_RATING_COL = "rating"
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_PREDICTION_COL = "prediction"
DEFAULT_HEADER = (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)
DEFAULT_SPLIT_FLAG = "split_flag"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.2
DEFAULT_DATA_LOCATION = "../Datasets/train_triplets/train_triplets.txt"
SMALL_DATA_LOCATION = "../Datasets/train_triplets/train_triplets_small.txt"
TINY_DATA_LOCATION = "../Datasets/train_triplets/train_triplets_tiny.txt"
DEFAULT_FMA_METADATA_LOCATION = "../Datasets/fma_metadata"
DEFAULT_FMA_SONG_LOCATION = "../Datasets/fma_small"
MILLION_SONG_CSV_LOCATION = "../Datasets/million_song_data.csv"