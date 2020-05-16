import pandas as pd
import multiprocessing
import functools
from joblib import Parallel, delayed
import sklearn.preprocessing as pp
from tqdm.notebook import tqdm
from recommender import item_based, tools as rtools, preprocessing as rpp
from recommender import item_based, dummy as dummy_recommender, preprocessing as rpp, model_selection as rms, \
    tools as rtools
import tag_genome_builder as tg_builder

from lib import tools
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


def evaluate_item_based(df_ratings: pd.DataFrame, df_item_feature: pd.DataFrame,
                        prediction_column_suffix: str,
                        save_prediction_path: str,
                        threshold: float = .6,
                        test_size: float = .25) -> pd.DataFrame:
    df_ratings_filtered = df_ratings[df_ratings[config.movieId_col].isin(df_item_feature.index)]
    user_activities = df_ratings_filtered[config.userId_col].value_counts()
    df_ratings_filtered = df_ratings_filtered[df_ratings_filtered[config.userId_col].
        isin(user_activities[user_activities > 1].index)]
    df_rating_train, df_rating_test = \
        rms.train_test_split(df_ratings_filtered, item_features=None, strategy='sklearn', test_size=test_size)
    vf_normalizer = tg_builder.VisualFeatureNormalizer()
    normalizer = pp.StandardScaler()
    df_item_feature_train = df_item_feature.loc[df_rating_train[config.movieId_col].unique()]
    df_agg_train_normalized = vf_normalizer.fit_transform(df_item_feature_train, normalizer)
    df_item_feature_test = df_item_feature.loc[df_rating_test[config.movieId_col].unique()]
    df_agg_test_normalized = vf_normalizer.transform(df_item_feature_test)

    item_features_w2v_train = rpp.ItemFeature()
    item_features_w2v_train.from_dataframe(df_agg_train_normalized)
    item_features_w2v_test = rpp.ItemFeature()
    item_features_w2v_test.from_dataframe(df_agg_test_normalized)
    recommend = item_based.ItemBasedColabCos()
    recommend.fit(df_rating_train, item_features_w2v_train)
    test_users = df_rating_test[config.userId_col].unique()
    recommendations_w2v = recommend.predict_on_list_of_users(test_users,
                                                             df_rating_test,
                                                             item_features_w2v_test,
                                                             n_jobs=7,
                                                             min_similarity=threshold)

    df_rating_test = rtools.prepare_recommendations_df(df_rating_test=df_rating_test,
                                                       recommendations=recommendations_w2v,
                                                       prediction_column_suffix=prediction_column_suffix)
    df_rating_test.to_csv(save_prediction_path)
    return tools.performance_report(df_rating_test, prediction_column_suffix=prediction_column_suffix)