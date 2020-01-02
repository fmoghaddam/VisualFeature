import sklearn.model_selection as ms
import pandas as pd
from recommender import base
import config


def my_train_test_split(df_rating: pd.DataFrame,
                        strategy='sklearn',
                        **kwargs) -> (pd.DataFrame, pd.DataFrame):
    implemented_strategies = ['last_n', 'sklearn']
    if strategy not in implemented_strategies:
        raise NotImplementedError(f'strategy {strategy} not known. Use one of {implemented_strategies}')
    if strategy == 'sklearn':
        return ms.train_test_split(df_rating, **kwargs)
    if strategy == 'last_n':
        return _split_last_n_movies_per_user(df_rating, **kwargs)


def _split_last_n_movies_per_user(df_rating: pd.DataFrame, number_of_items_in_test_per_user: int):
    df_rating.sort_values(config.timestamp_col, inplace=True)
    df_rating_test = df_rating.groupby(config.userId_col).tail(number_of_items_in_test_per_user)
    df_rating_train = df_rating.drop(df_rating_test.index)
    return df_rating_train, df_rating_test


def train_test_split(df_rating: pd.DataFrame, item_features: base.ItemFeature, strategy, **kwargs) -> object:
    # df_rating.sort_values([config.timestamp_col])
    df_rating_train, df_rating_test = \
        my_train_test_split(df_rating, stratify=df_rating[config.userId_col],
                            strategy=strategy, **kwargs)
    item_features_train =\
        item_features.get_item_feature_by_list_of_items(df_rating_train[config.movieId_col].unique())
    item_features_test =\
        item_features.get_item_feature_by_list_of_items(df_rating_test[config.movieId_col].unique())
    return df_rating_train, df_rating_test, item_features_train, item_features_test
