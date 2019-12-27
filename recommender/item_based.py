from scipy import sparse
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
        self.item_ids = item_ids
        self.feature_names = feature_names
        self.feature_matrix = feature_matrix


class ItemBasedColabCos(object):
    def __init__(self):
        pass

    def fit(self, df_rating: pd.DataFrame, item_features: ItemFeature):
        """make a dictionary {user_id, (list of rated movies, np.array of respective rates)"""
        assert set(df_rating.columns).issubset(movie_rating_cols), ('df_rating has to have at least these '
                                                                    f'columns: {movie_rating_cols}')

        self.item_features = item_features
        movie_rating = namedtuple('movie_rating', f'{movieId_col} {rating_col}')
        df_fit = df_rating.groupby(userId_col).agg({movieId_col: lambda x: x.tolist(),
                                                    rating_col: lambda x: x.tolist()})
        self.dict_user_ratings = dict(df_fit.apply(lambda row:
                                                   movie_rating(row[movieId_col], row[rating_col]), axis=1))

    def predict(self, user_id, new_items):
        """for the given user_id give the predicted rates for new_items"""
        check_is_fitted(self, ['item_features', 'dict_user_ratings'])
        csr_user_matrix = self.get_user_matrix(user_id)
        l_user_ratings = self.dict_user_ratings.get(user_id)
        assert l_user_ratings is not None, 'user not found'
        csr_new_items_matrix = self.get_items_matrix(new_items)
        csr_similarities = csr_new_items_matrix.dot(csr_user_matrix.transpose())
        csr_similarities_weighted = sparse.diags(l_user_ratings).dot(csr_similarities)
        new_ratings = csr_similarities_weighted.sum(axis=1) / csr_similarities.sum(axis=1)

    def items_to_feature_space(self, df) -> sparse.csr_matrix:
        pass
