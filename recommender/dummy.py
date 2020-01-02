import numpy as np
import pandas as pd

from recommender.base import ItemFeature
from recommender import base
import config


class DummyAverageUser(base.BaseRecommender):

    def fit(self, df_rating: pd.DataFrame, item_features: ItemFeature):
        self._validate_fit_input(df_rating, item_features)
        self.item_features = item_features
        df_fit = df_rating.groupby(config.userId_col)[[config.rating_col]].agg('mean')
        self.dict_user_ratings = dict(zip(df_fit.index, df_fit[config.rating_col]))
        return self

    def predict(self, user_id: int, new_items: ItemFeature) -> pd.DataFrame:
        self._validate_predict_input(user_id, new_items)
        this_user_average_rating = self.dict_user_ratings.get(user_id, np.nan)
        new_ratings = [this_user_average_rating] * len(new_items)
        return self._prepare_prediction_output(new_items, new_ratings)
