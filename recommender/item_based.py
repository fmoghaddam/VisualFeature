import time
from scipy import sparse
import numpy as np
import pandas as pd
from collections import namedtuple
import multiprocessing
import functools

from .base import ItemFeature
from lib import check_is_fitted, tools
import config


movieId_col = config.movieId_col
userId_col = config.userId_col
rating_col = config.rating_col
movie_rating_cols = [movieId_col, userId_col, rating_col]
movie_rating = namedtuple('movie_rating', f'{movieId_col} {rating_col}')

class ItemBasedColabCos(object):
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

    def _validate_fit_input(self, df_rating: pd.DataFrame, item_features: ItemFeature):
        if not isinstance(df_rating, pd.DataFrame):
            raise TypeError('Only pandas DataFrame are accepted as input for rating')
        if not isinstance(item_features, ItemFeature):
            raise TypeError(f'new items has to be of type ItemFeature. It is of type {type(item_features)}')
        assert set(movie_rating_cols).issubset(df_rating.columns), ('df_rating has to have at least these '
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
        predicted_ratings_with_item_id_for_user_id = pd.DataFrame({f'{rating_col}_predicted': new_ratings}, index=index)
        return predicted_ratings_with_item_id_for_user_id

    def get_new_ratings(self, csr_new_items_matrix, csr_user_matrix, l_user_ratings):
        csr_similarities = csr_new_items_matrix.dot(csr_user_matrix.transpose())
        csr_similarities_weighted = csr_similarities.dot(sparse.diags(l_user_ratings))
        new_ratings = csr_similarities_weighted.sum(axis=1) / csr_similarities.sum(axis=1)
        new_ratings_flat = np.array(new_ratings).ravel()
        return new_ratings_flat

    def predict_on_list_of_users(self, users, df_rating_test, item_features, n_jobs=1):
        valid_users = set(users).intersection(self.dict_user_ratings.keys())
        if len(valid_users) < len(users):
            print(f'Warning: {len(users) - len(valid_users)} users were not valid')
        if len(valid_users) == 0:
            return None
        if n_jobs == 1:
            return self.predict_on_list_of_users_single_job(valid_users, df_rating_test, item_features)
        else:
            return self.predict_on_list_of_users_parallel(valid_users, df_rating_test, item_features,
                                                          n_jobs=n_jobs)

    def predict_on_list_of_users_single_job(self, users, df_rating_test, item_features):
        l_preds = []
        t0 = time.time()
        for i, user in enumerate(users):
            if i % (len(users) // 50) == 0:
                tools.update_progress(i / len(users), t0)
            _pred = self._loop(df_rating_test, item_features, user)
            l_preds.append(_pred)
        output = pd.concat(l_preds, ignore_index=True)
        return output

    def predict_on_list_of_users_parallel(self, users, df_rating_test, item_features, n_jobs):
        loop = functools.partial(self._loop, df_rating_test, item_features)
        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count() + n_jobs + 1
        pool = multiprocessing.Pool(n_jobs)
        try:
            l_preds = pool.map(loop, users)
        except Exception as e:
            print(e)
            l_preds = [pd.DataFrame()]
            raise
        finally:
            pool.close()
            output = pd.concat(l_preds, ignore_index=True)
        return output

    def _loop(self, df_rating_test, item_features, user):
        _movie_ids = df_rating_test.loc[
            df_rating_test[config.userId_col] == user, config.movieId_col].tolist()
        new_items = item_features.get_item_feature_by_list_of_items(_movie_ids)
        _pred = self.predict(user, new_items)
        _pred[config.userId_col] = user
        return _pred.reset_index()[[config.userId_col, config.movieId_col, f'{config.rating_col}_predicted']]

    def items_to_feature_space(self, df) -> sparse.csr_matrix:
        pass
