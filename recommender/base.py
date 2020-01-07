from scipy import sparse
import numpy as np
import pandas as pd
import multiprocessing
import functools
import time
from lib import check_is_fitted, tools
import config
movieId_col = config.movieId_col
userId_col = config.userId_col
rating_col = config.rating_col
movie_rating_cols = [movieId_col, userId_col, rating_col]


class ItemFeature(object):
    def __init__(self, item_ids: list = None, feature_names=None,
                 feature_matrix: sparse.csr_matrix = None):
        if item_ids is not None:
            self._initiate(item_ids, feature_names, feature_matrix)

    def __len__(self):
        return self.shape[0]

    def _initiate(self, item_ids: list, feature_names: list,
                  feature_matrix: sparse.csr_matrix):
        self._validate_input(item_ids, feature_names, feature_matrix)
        self.item_ids = np.array(item_ids)
        self.feature_names = np.array(feature_names)
        self.feature_matrix = feature_matrix
        self.shape = (len(item_ids), len(feature_names))

    def _validate_input(self, item_ids: list, feature_names: list,
                        feature_matrix: sparse.csr_matrix):
        if not isinstance(feature_matrix, sparse.csr_matrix):
            raise TypeError('only sparse.csr_matrix can be accepted as feature matrix')
        assert feature_matrix.shape == (len(item_ids), len(feature_names)), ('dimension mismatch, '
                                                                             'feature_matrix does not have '
                                                                             'compatible shape comparing to '
                                                                             'number of items and number of '
                                                                             'features')
        no_of_nulls_in_feature_matrix = pd.isnull(feature_matrix.data).sum()
        if no_of_nulls_in_feature_matrix > 0:
            raise ValueError(f'feature matrix contains {no_of_nulls_in_feature_matrix}'
                             f' missing values. Do something about them first')

    def from_dataframe(self, df: pd.DataFrame):
        """df has movieId's as index and feature names as columns"""
        self._initiate(df.index, df.columns, sparse.csr_matrix(df.values))

    def get_feature_matrix_by_list_of_items(self, some_item_ids):
        assert set(some_item_ids).issubset(self.item_ids), 'I do not have all the items you wanted'
        item_ids_indices = np.array([np.where(self.item_ids == item)[0][0]
                                     for item in some_item_ids
                                     if item in self.item_ids])
        return self.feature_matrix[item_ids_indices, :]

    def get_item_feature_by_list_of_items(self, some_item_ids):
        return ItemFeature(item_ids=some_item_ids,
                           feature_names=self.feature_names,
                           feature_matrix=self.get_feature_matrix_by_list_of_items(some_item_ids))

    def to_dataframe(self):
        df = pd.DataFrame(self.feature_matrix.toarray(),
                          index=self.item_ids,
                          columns=self.feature_names)
        df.index.name = config.movieId_col
        return df


class BaseRecommender(object):

    def _validate_fit_input(self, df_rating: pd.DataFrame, item_features: ItemFeature):
        if not isinstance(df_rating, pd.DataFrame):
            raise TypeError('Only pandas DataFrame are accepted as input for rating')
        if not isinstance(item_features, ItemFeature):
            raise TypeError(f'new items has to be of type ItemFeature. It is of type {type(item_features)}')
        assert set(movie_rating_cols).issubset(df_rating.columns), ('df_rating has to have at least these '
                                                                    f'columns: {movie_rating_cols}')

    def _validate_predict_input(self, user_id: int, new_items: ItemFeature):
        check_is_fitted(self, ['item_features', 'dict_user_ratings'])
        assert (new_items.feature_names == self.item_features.feature_names).all(), ('The feature set in the '
                                                                                     'new items is not '
                                                                                     'identical with the '
                                                                                     'feature set used for '
                                                                                     'fitting')
        assert user_id in self.dict_user_ratings.keys(), 'unknown user'

    def _prepare_prediction_output(self, new_items: ItemFeature, new_ratings: np.array) -> pd.DataFrame:
        index = pd.Index(new_items.item_ids, name=movieId_col)
        predicted_ratings_with_item_id_for_user_id = pd.DataFrame({f'{rating_col}_predicted': new_ratings},
                                                                  index=index)
        return predicted_ratings_with_item_id_for_user_id

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
