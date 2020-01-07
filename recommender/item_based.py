from scipy import sparse
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn import metrics
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
        similarities = metrics.pairwise.cosine_similarity(csr_new_items_matrix, csr_user_matrix,
                                                          dense_output=True)
        similarities_weighted = similarities.dot(np.diag(l_user_ratings))
        new_ratings = similarities_weighted.sum(axis=1) / abs(similarities).sum(axis=1)
        new_ratings_flat = np.array(new_ratings).ravel()
        return new_ratings_flat

    def items_to_feature_space(self, df) -> sparse.csr_matrix:
        pass


class ItemBasedColabCos2(base.BaseRecommender):
    def __init__(self):
        pass

    def fit(self, df_rating: pd.DataFrame, item_features: ItemFeature):
        """make a dictionary {user_id, (list of rated movies, np.array of respective rates)"""
        self._validate_fit_input(df_rating, item_features)
        self.item_features = item_features
        self.df_rating = df_rating[movie_rating_cols]
        self.dict_user_ratings = dict(df_rating[config.userId_col].value_counts())
        return self

    def predict(self, user_id: int, new_items: ItemFeature) -> pd.DataFrame:
        """for the given user_id give the predicted rates for new_items"""
        self._validate_predict_input(user_id, new_items)

        csr_new_items_matrix = self.get_items_matrix(new_items)
        csr_user_matrix = self.get_user_matrix([user_id])
        l_user_ratings = self.get_user_ratings([user_id])
        new_ratings = self.get_new_ratings(csr_new_items_matrix, csr_user_matrix, l_user_ratings)
        return self._prepare_prediction_output(new_items, new_ratings)

    def get_user_ratings(self, user_id):
        return self.df_rating.loc[self.df_rating[config.userId_col].isin(user_id), config.rating_col].values

    def get_user_matrix(self, user_id):
        item_ids_rated_by_user = self.df_rating.loc[self.df_rating[config.userId_col].isin(user_id),
                                                    config.movieId_col].values
        return self.item_features.get_feature_matrix_by_list_of_items(item_ids_rated_by_user)

    def get_items_matrix(self, new_items: ItemFeature):
        return new_items.feature_matrix

    def get_new_ratings(self, csr_new_items_matrix, csr_user_matrix, l_user_ratings):
        similarities = metrics.pairwise.cosine_similarity(csr_new_items_matrix, csr_user_matrix,
                                                          dense_output=True)
        similarities_weighted = similarities.dot(np.diag(l_user_ratings))
        new_ratings = similarities_weighted.sum(axis=1) / abs(similarities).sum(axis=1)
        new_ratings_flat = np.array(new_ratings).ravel()
        return new_ratings_flat

    def predict_on_list_of_users_vectorize(self, users, new_items):
        # self._validate_predict_input(users, new_items)

        csr_new_items_matrix = self.get_items_matrix(new_items)
        csr_user_matrix = self.get_user_matrix(users)
        l_user_ratings = self.get_user_ratings(users)
        l_repeated_users = self.get_user_ids_repeated(users)
        new_ratings = self.get_new_ratings_list(csr_new_items_matrix, csr_user_matrix, l_user_ratings,
                                                l_repeated_users, new_items)
        return new_ratings

    def get_user_ids_repeated(self, user_id):
        return self.df_rating.loc[self.df_rating[config.userId_col].isin(user_id), config.userId_col].values

    def get_new_ratings_list(self, csr_new_items_matrix, csr_user_matrix, l_user_ratings,
                             l_repeated_users, new_items):
        similarities = metrics.pairwise.cosine_similarity(csr_user_matrix, csr_new_items_matrix,
                                                          dense_output=True)
        similarities_weighted = np.diag(l_user_ratings).dot(similarities)
        user_summed_similarities = pd.DataFrame(similarities, columns=new_items.item_ids).groupby(
            l_repeated_users).agg('sum')
        user_summed_similarities_weighted = pd.DataFrame(similarities_weighted,
                                                         columns=new_items.item_ids).groupby(
            l_repeated_users).agg('sum')
        new_ratings = user_summed_similarities_weighted / user_summed_similarities.abs()
        new_ratings.index.name = config.userId_col
        # df_new_ratings = pd.melt(new_ratings.reset_index(), id_vars=[config.userId_col],
        #                          var_name=config.movieId_col,
        #                          value_name=f'{config.rating_col}_predicted')
        return new_ratings

    def items_to_feature_space(self, df) -> sparse.csr_matrix:
        pass
