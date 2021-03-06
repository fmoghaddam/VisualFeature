import numpy as np
import pandas as pd
from scipy import sparse
from recommender import item_based
from recommender import base
import config

m = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
csr_m = sparse.csr_matrix(m)
item_features = base.ItemFeature(item_ids=[1, 2, 3, 4],
                                 feature_names=['f1', 'f2', 'f3', 'f4'],
                                 feature_matrix=csr_m)

m2 = np.array([[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5], [9.5, 10.5, 11.5, 12.5]])
csr_m2 = sparse.csr_matrix(m2)
item_features_new = base.ItemFeature(item_ids=[8, 9, 10],
                                     feature_names=['f1', 'f2', 'f3', 'f4'],
                                     feature_matrix=csr_m2)

df_rating = pd.DataFrame([[1, 2, .3], [1, 4, 4.5], [2, 2, 5]],
                         columns=[item_based.userId_col, item_based.movieId_col, item_based.rating_col])

item_based_colab_cos = item_based.ItemBasedColabCos2()

item_based_colab_cos.fit(df_rating=df_rating,
                         item_features=item_features)


def test_fit_dict_user_ratings():
    expected = {1: 2, 2: 1}
    actual = item_based_colab_cos.dict_user_ratings
    assert actual == expected


def test_get_user_matrix():
    expected = np.array([[5, 6, 7, 8], [13, 14, 15, 16]])
    user_id = [1]
    actual = item_based_colab_cos.get_user_matrix(user_id).toarray()
    assert len(actual) == len(expected)
    assert (actual == expected).all()


def test_get_user_matrix_list():
    expected = np.array([[5, 6, 7, 8], [13, 14, 15, 16], [5, 6, 7, 8]])
    user_id = [1, 2]
    actual = item_based_colab_cos.get_user_matrix(user_id).toarray()
    assert len(actual) == len(expected)
    assert (actual == expected).all()


def test_get_user_ratings():
    expected = np.array([.3, 4.5])
    user_id = [1]
    actual = item_based_colab_cos.get_user_ratings(user_id)
    assert len(actual) == len(expected)
    assert (actual == expected).all()


def test_get_user_ratings_list():
    expected = np.array([.3, 4.5, 5])
    user_id = [1, 2]
    actual = item_based_colab_cos.get_user_ratings(user_id)
    assert len(actual) == len(expected)
    assert (actual == expected).all()


def test_get_items_matrix():
    expected = np.array([[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5], [9.5, 10.5, 11.5, 12.5]])
    actual = item_based_colab_cos.get_items_matrix(item_features_new).toarray()
    assert len(actual) == len(expected)
    assert np.allclose(actual, expected, 1e-5)


def test_get_new_ratings():
    new_items_matrix = np.array([[1.5, 2.5, 3.5, 4.5], [5.5, 6.5, 7.5, 8.5], [9.5, 10.5, 11.5, 12.5]])
    csr_new_items_matrix = sparse.csr_matrix(new_items_matrix)
    user_matrix = np.array([[5, 6, 7, 8], [13, 14, 15, 16]])
    csr_user_matrix = sparse.csr_matrix(user_matrix)
    l_user_ratings = [.3, 4.5]
    expected = np.array([2.376704781832865, 2.396590198814686, 2.4021938974156347])
    actual = item_based_colab_cos.get_new_ratings(csr_new_items_matrix, csr_user_matrix, l_user_ratings)
    assert len(actual) == len(expected)
    assert np.allclose(actual, expected, 1e-5)


def test_predict():
    index = pd.Index([8, 9, 10], name=item_based.movieId_col)
    expected = pd.DataFrame({f'{item_based.rating_col}_predicted':
                            np.array([2.376704781832865, 2.396590198814686, 2.4021938974156347])},
                            index=index)
    actual = item_based_colab_cos.predict(1, item_features_new)
    assert expected.columns == actual.columns
    assert expected.shape == actual.shape
    assert np.allclose(expected.values, actual.values, 1e-5)


def test_predict_on_list_of_users_vectorize():
    list_predicted = {8: {"1": 2.3767047818, "2": 5.0},
                      9: {"1": 2.3965901988, "2": 5.0},
                      10: {"1": 2.4021938974, "2": 5.0}}
    expected = pd.DataFrame(list_predicted)
    actual = item_based_colab_cos.predict_on_list_of_users_vectorize([1, 2], item_features_new)
    assert (expected.columns == actual.columns).all()
    assert actual.index.name == config.userId_col
    assert np.allclose(actual, expected)
