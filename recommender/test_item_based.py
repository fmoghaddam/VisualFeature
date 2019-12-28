import numpy as np
import pandas as pd
from scipy import sparse
from recommender import item_based

m = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
csr_m = sparse.csr_matrix(m)
item_features = item_based.ItemFeature(item_ids=[1, 2, 3, 4],
                                       feature_names=['f1', 'f2', 'f3', 'f4'],
                                       feature_matrix=csr_m)

m2 = np.array([[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5], [9.5, 10.5, 11.5, 12.5]])
csr_m2 = sparse.csr_matrix(m2)
item_features_new = item_based.ItemFeature(item_ids=[8, 9, 10],
                                           feature_names=['f1', 'f2', 'f3', 'f4'],
                                           feature_matrix=csr_m2)

item_based_colab_cos = item_based.ItemBasedColabCos()
df_rating = pd.DataFrame([[1, 2, .3], [1, 4, 4.5]],
                         columns=[item_based.userId_col, item_based.movieId_col, item_based.rating_col])
item_based_colab_cos.fit(df_rating=df_rating,
                         item_features=item_features)


def test_fit_dict_user_ratings():
    expected = {1: ([2, 4], [.3, 4.5])}
    actual = item_based_colab_cos.dict_user_ratings
    assert actual == expected


def test_get_user_matrix():
    expected = [[5, 6, 7, 8], [13, 14, 15, 16]]
    actual = item_based_colab_cos.get_user_matrix(1).toarray()
    assert len(actual) == len(expected)
    assert (actual == expected).all(), f'{actual}'


def test_get_items_matrix():
    expected = np.array([[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5], [9.5, 10.5, 11.5, 12.5]])
    actual = item_based_colab_cos.get_items_matrix(item_features_new).toarray()
    assert len(actual) == len(expected)
    assert (actual == expected).all(), f'{actual}'
