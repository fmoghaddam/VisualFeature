import multiprocessing
import functools
from joblib import Parallel, delayed
from recommender import item_based, tools as rtools, preprocessing as rpp
import config


def get_predictions_for_different_number_of_tags_loop(df_predicted_tag_genome,
                                                      df_rating_train,
                                                      df_rating_test,
                                                      number_of_tag_per_movie):
    prediction_column_suffix = f'vf_tg_{number_of_tag_per_movie}'
    item_features_vf_tg = rpp.get_item_feature_from_tag_genome(df_predicted_tag_genome,
                                                               number_of_tag_per_movie)
    item_features_train = \
        item_features_vf_tg.get_item_feature_by_list_of_items(df_rating_train[config.movieId_col].unique())
    item_features_test = \
        item_features_vf_tg.get_item_feature_by_list_of_items(df_rating_test[config.movieId_col].unique())
    recommend = item_based.ItemBasedColabCos()
    recommend.fit(df_rating_train, item_features_train)
    test_users = df_rating_test[config.userId_col].unique()
    recommendations = recommend.predict_on_list_of_users(test_users, df_rating_test, item_features_test,
                                                         n_jobs=1)
    return [recommendations, prediction_column_suffix]


def get_predictions_for_different_number_of_tags(df_predicted_tag_genome,
                                                 steps,
                                                 df_rating_train,
                                                 df_rating_test,
                                                 n_jobs=-2):
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count() + n_jobs + 1
    loop = functools.partial(get_predictions_for_different_number_of_tags_loop,
                             df_predicted_tag_genome,
                             df_rating_train,
                             df_rating_test
                             )
    recs = Parallel(n_jobs=n_jobs, verbose=30)(
        delayed(loop)(number_of_tag_per_movie)
        for number_of_tag_per_movie in steps
    )
    for recommendations, prediction_column_suffix in recs:
        df_rating_test = rtools.prepare_recommendations_df(df_rating_test=df_rating_test,
                                                           recommendations=recommendations,
                                                           prediction_column_suffix=prediction_column_suffix)
    return df_rating_test
