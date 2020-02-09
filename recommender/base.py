from scipy import sparse
import numpy as np
import pandas as pd
import multiprocessing
import functools
from tqdm import tqdm

import time
from lib import check_is_fitted, tools
import config
from base.data_types import ItemFeature
movieId_col = config.movieId_col
userId_col = config.userId_col
rating_col = config.rating_col
movie_rating_cols = [movieId_col, userId_col, rating_col]


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

    def predict_on_list_of_users(self, users, df_rating_test, item_features, n_jobs=1, min_similarity=0):
        valid_users = set(users).intersection(self.dict_user_ratings.keys())
        if len(valid_users) < len(users):
            print(f'Warning: {len(users) - len(valid_users)} users were not valid')
        if len(valid_users) == 0:
            return None
        if n_jobs == 1:
            return self.predict_on_list_of_users_single_job(valid_users, df_rating_test, item_features,
                                                            min_similarity=min_similarity)
        else:
            return self.predict_on_list_of_users_parallel(valid_users, df_rating_test, item_features,
                                                          n_jobs=n_jobs,
                                                          min_similarity=min_similarity)

    def predict_on_list_of_users_single_job(self, users, df_rating_test, item_features, min_similarity):
        l_preds = []
        t0 = time.time()
        for i, user in tqdm(enumerate(users), total=len(users)):
            # if i % (len(users) // 50) == 0:
            #     tools.update_progress(i / len(users), t0)
            _pred = self._loop(df_rating_test, item_features, user, min_similarity=min_similarity)
            l_preds.append(_pred)
        output = pd.concat(l_preds, ignore_index=True)
        return output

    def predict_on_list_of_users_parallel(self, users, df_rating_test, item_features, n_jobs, min_similarity):
        number_of_all_users = len(users)
        loop = functools.partial(self._loop, df_rating_test, item_features, min_similarity=min_similarity)
        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count() + n_jobs + 1
        pool = multiprocessing.Pool(n_jobs)
        l_preds = []
        try:
            # for pred in tqdm(pool.imap(loop, users), total=number_of_all_users):
            #     l_preds.append(pred)
            l_preds = pool.map(loop, users)
        except Exception as e:
            print(e)
            l_preds = [pd.DataFrame()]
            raise
        finally:
            pool.close()
            output = pd.concat(l_preds, ignore_index=True)
        return output

    def _loop(self, df_rating_test, item_features, user, min_similarity):
        _movie_ids = df_rating_test.loc[
            df_rating_test[config.userId_col] == user, config.movieId_col].tolist()
        new_items = item_features.get_item_feature_by_list_of_items(_movie_ids)
        _pred = self.predict(user, new_items, min_similarity=min_similarity)
        _pred[config.userId_col] = user
        return _pred.reset_index()[[config.userId_col, config.movieId_col, f'{config.rating_col}_predicted']]
