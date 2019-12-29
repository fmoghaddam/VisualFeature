import sklearn.model_selection as ms
import recommender.item_based as settings  # FIXME change it later to config
from recommender import item_based
import pandas as pd


def train_test_split(df_rating: pd.DataFrame, item_features: item_based.ItemFeature) -> object:
    df_rating_train, df_rating_test = \
        ms.train_test_split(df_rating, stratify=df_rating[settings.userId_col])
    item_features_train =\
        item_features.get_feature_matrix_by_list_of_items(df_rating_train[settings.movieId_col].unique())
    item_features_test =\
        item_features.get_feature_matrix_by_list_of_items(df_rating_test[settings.movieId_col].unique())
    return df_rating_train, df_rating_test, item_features_train, item_features_test
