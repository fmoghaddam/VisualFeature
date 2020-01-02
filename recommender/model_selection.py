import sklearn.model_selection as ms
import recommender.item_based as settings  # FIXME change it later to config
from recommender import base
import pandas as pd
import config


def my_train_test_split(df_rating: pd.DataFrame,
                        strategy='sklearn',
                        number_of_items_in_test_per_user=5,
                        **kwargs) -> (pd.DataFrame, pd.DataFrame):
    implemented_strategies = ['last_n', 'sklearn']
    if strategy not in implemented_strategies:
        raise NotImplementedError(f'strategy {strategy} not known. Use one of {implemented_strategies}')
    if strategy == 'sklearn':
        return ms.train_test_split(df_rating, **kwargs)


def train_test_split(df_rating: pd.DataFrame, item_features: base.ItemFeature, strategy, **kwargs) -> object:
    # df_rating.sort_values([config.timestamp_col])
    df_rating_train, df_rating_test = \
        my_train_test_split(df_rating, stratify=df_rating[settings.userId_col],
                            strategy=strategy, **kwargs)
    item_features_train =\
        item_features.get_item_feature_by_list_of_items(df_rating_train[settings.movieId_col].unique())
    item_features_test =\
        item_features.get_item_feature_by_list_of_items(df_rating_test[settings.movieId_col].unique())
    return df_rating_train, df_rating_test, item_features_train, item_features_test
