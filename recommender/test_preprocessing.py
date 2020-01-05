import numpy as np
import pandas as pd
import config
import recommender.preprocessing as rpp

df_top_n_sample = pd.DataFrame({config.tagId_col: '786|785|589|588|536|244|204|186|64|63'},
                               index=pd.Index([1], name=config.movieId_col))
sample_rating = pd.DataFrame({config.userId_col: [1, 1, 1, 2, 2, 1, 2, 1, 2],
                              config.movieId_col: np.arange(9),
                              config.rating_col: [3.5, 4, 4.5, 1, 2, 5, 2.5, 3, 3]
                              })


def test_get_item_feature_from_tag_genome():
    # TODO mock get_top_n_tags to return df_top_n_sample
    expected_items_ids = np.array([1])
    expected_feature_names = np.array(['186', '204', '244', '536', '588', '589', '63', '64', '785', '786'])
    expected_feature_matrix_array = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


def test_rating_normalizer_fit_transform():
    expected = np.array([-0.5, 0., 0.5, -1.125, -0.125, 1., 0.375, -1., 0.875])
    rating_normalizer = rpp.RatingNormalizer()
    scaled_ratings = rating_normalizer.fit_transform(sample_rating)
    assert np.allclose(scaled_ratings, expected, 1e-5)


def test_rating_normalizer_transform():
    rating_normalizer = rpp.RatingNormalizer()
    rating_normalizer.fit(sample_rating.loc[0:6, :])
    assert np.allclose(np.array([-1.25, 1.16666667]), rating_normalizer.transform(sample_rating.loc[7:]))
