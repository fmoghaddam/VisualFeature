from scipy import sparse
import numpy as np
import pandas as pd
from collections import namedtuple

from .base import ItemFeature
from recommender import base
import config


movieId_col = config.movieId_col
userId_col = config.userId_col
rating_col = config.rating_col
movie_rating_cols = [movieId_col, userId_col, rating_col]
movie_rating = namedtuple('movie_rating', f'{movieId_col} {rating_col}')


class ItemBasedColabCos(base.BaseRecommender):
    def __init__(self):
        pass

    def fit(self, df_rating: pd.DataFrame, item_features: ItemFeature):
        """make a dictionary {user_id, (list of rated movies, np.array of respective rates)"""
        self._validate_fit_input(df_rating, item_features)
        self.item_features = item_features

        df_fit = df_rating.groupby(userId_col).agg({movieId_col: lambda x: x.tolist(),
                                                    rating_col: lambda x: x.tolist()})
        self.dict_user_ratings = dict(df_fit.apply(lambda row:
                                                   movie_rating(row[movieId_col], row[rating_col]), axis=1))
        return self

    def predict(self, user_id: int, new_items: ItemFeature) -> pd.DataFrame:
        """for the given user_id give the predicted rates for new_items"""
        self._validate_predict_input(user_id, new_items)

        csr_new_items_matrix = self.get_items_matrix(new_items)
        user_info = self.dict_user_ratings.get(user_id)
        csr_user_matrix = self.get_user_matrix(user_info)
        l_user_ratings = getattr(user_info, rating_col)
        new_ratings = self.get_new_ratings(csr_new_items_matrix, csr_user_matrix, l_user_ratings)
        return self._prepare_prediction_output(new_items, new_ratings)

    def get_user_matrix(self, user_info):
        item_ids_rated_by_user = getattr(user_info, movieId_col)
        return self.item_features.get_feature_matrix_by_list_of_items(item_ids_rated_by_user)

    def get_items_matrix(self, new_items: ItemFeature):
        return new_items.feature_matrix

    def get_new_ratings(self, csr_new_items_matrix, csr_user_matrix, l_user_ratings):
        csr_similarities = csr_new_items_matrix.dot(csr_user_matrix.transpose())
        csr_similarities_weighted = csr_similarities.dot(sparse.diags(l_user_ratings))
        new_ratings = csr_similarities_weighted.sum(axis=1) / abs(csr_similarities).sum(axis=1)
        new_ratings_flat = np.array(new_ratings).ravel()
        return new_ratings_flat

    def items_to_feature_space(self, df) -> sparse.csr_matrix:
        pass
