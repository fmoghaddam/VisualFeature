from scipy import sparse
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.preprocessing.label import check_is_fitted
# import sklearn.preprocessing as pp

movieId_col = 'movieId'
userId_col = 'userId'
rating_col = 'rating'
movie_rating_cols = [movieId_col, userId_col, rating_col]


class ItemFeature(object):
    def __init__(self, item_ids: list, feature_names: list,
                 feature_matrix: sparse.csr_matrix):
        assert isinstance(feature_matrix, sparse.csr_matrix), ('only sparse.csr_matrix can be accepted as '
                                                               'feature matrix')
        assert feature_matrix.shape == (len(item_ids), len(feature_names)), ('dimension mismatch, '
                                                                             'feature_matrix does not have '
                                                                             'compatible shape comparing to '
                                                                             'number of items and number of '
                                                                             'features')
        self.item_ids = np.array(item_ids)
        self.feature_names = np.array(feature_names)
        self.feature_matrix = feature_matrix

    def get_feature_matrix_by_list_of_items(self, some_item_ids):
        item_ids_indices = np.array([np.where(self.item_ids == item)[0][0]
                                     for item in some_item_ids
                                     if item in self.item_ids])
        return self.feature_matrix[item_ids_indices, :]


class ItemBasedColabCos(object):
    def __init__(self):
        pass

    def fit(self, df_rating: pd.DataFrame, item_features: ItemFeature):
        """make a dictionary {user_id, (list of rated movies, np.array of respective rates)"""
        self._validate_fit_input(df_rating, item_features)
        self.item_features = item_features
        movie_rating = namedtuple('movie_rating', f'{movieId_col} {rating_col}')
        df_fit = df_rating.groupby(userId_col).agg({movieId_col: lambda x: x.tolist(),
                                                    rating_col: lambda x: x.tolist()})
        self.dict_user_ratings = dict(df_fit.apply(lambda row:
                                                   movie_rating(row[movieId_col], row[rating_col]), axis=1))

    def _validate_fit_input(self, df_rating: pd.DataFrame, item_features: ItemFeature):
        if not isinstance(df_rating, pd.DataFrame):
            raise TypeError('Only pandas DataFrame are accepted as input for rating')
        if not isinstance(item_features, ItemFeature):
            raise TypeError('new items has to be of type ItemFeature')
        assert set(df_rating.columns).issubset(movie_rating_cols), ('df_rating has to have at least these '
                                                                    f'columns: {movie_rating_cols}')

    def predict(self, user_id: int, new_items: ItemFeature) -> pd.DataFrame:
        """for the given user_id give the predicted rates for new_items"""
        self._validate_predict_input(user_id, new_items)

        csr_new_items_matrix = self.get_items_matrix(new_items)
        user_info = self.dict_user_ratings.get(user_id)
        csr_user_matrix = self.get_user_matrix(user_info)
        l_user_ratings = getattr(user_info, rating_col)
        new_ratings = self.get_new_ratings(csr_new_items_matrix, csr_user_matrix, l_user_ratings)
        return self._prepare_prediction_output(new_items, new_ratings)

    def _validate_predict_input(self, user_id: int, new_items: ItemFeature):
        check_is_fitted(self, ['item_features', 'dict_user_ratings'])
        assert (new_items.feature_names == self.item_features.feature_names).all(), ('The feature set in the '
                                                                                     'new items is not '
                                                                                     'identical with the '
                                                                                     'feature set used for '
                                                                                     'fitting')
        assert user_id in self.dict_user_ratings.keys(), 'unknown user'

    def get_user_matrix(self, user_info):
        item_ids_rated_by_user = getattr(user_info, movieId_col)
        return self.item_features.get_feature_matrix_by_list_of_items(item_ids_rated_by_user)

    def get_items_matrix(self, new_items: ItemFeature):
        return new_items.feature_matrix

    def _prepare_prediction_output(self, new_items: ItemFeature, new_ratings: np.array) -> pd.DataFrame:
        index = pd.Index(new_items.item_ids, name=movieId_col)
        predicted_ratings_with_item_id_for_user_id = pd.DataFrame({rating_col: new_ratings}, index=index)
        return predicted_ratings_with_item_id_for_user_id

    def get_new_ratings(self, csr_new_items_matrix, csr_user_matrix, l_user_ratings):
        csr_similarities = csr_new_items_matrix.dot(csr_user_matrix.transpose())
        csr_similarities_weighted = csr_similarities.dot(sparse.diags(l_user_ratings))
        new_ratings = csr_similarities_weighted.sum(axis=1) / csr_similarities.sum(axis=1)
        new_ratings = np.array(new_ratings).ravel()
        return new_ratings

    def items_to_feature_space(self, df) -> sparse.csr_matrix:
        pass
