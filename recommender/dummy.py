import numpy as np
import pandas as pd

from recommender.base import ItemFeature
from recommender import base
import config


class DummyAverageUser(base.BaseRecommender):

    def fit(self, df_rating: pd.DataFrame, item_features: ItemFeature):
        self._validate_fit_input(df_rating, item_features)
        self.item_features = item_features
        self.df_fit = df_rating.groupby(config.userId_col)[[config.rating_col]].agg('mean')
        self.dict_user_ratings = dict(zip(self.df_fit.index, self.df_fit[config.rating_col]))
        return self

    def predict(self, user_id: int, new_items: ItemFeature) -> pd.DataFrame:
        self._validate_predict_input(user_id, new_items)
        this_user_average_rating = self.dict_user_ratings.get(user_id, np.nan)
        new_ratings = [this_user_average_rating] * len(new_items)
        return self._prepare_prediction_output(new_items, new_ratings)

    def predict_on_list_of_users(self, users, df_rating_test, item_features, n_jobs=1):
        output = df_rating_test[[config.userId_col, config.movieId_col]].copy()
        output[f'{config.rating_col}_predicted'] = self.df_fit.loc[df_rating_test[config.userId_col],
                                                                   config.rating_col].values
        return output
