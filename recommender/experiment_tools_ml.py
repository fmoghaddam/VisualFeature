import numpy as np
import pandas as pd
import os
from scipy import sparse
from joblib import Parallel, delayed
import sklearn.preprocessing as pp
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import base.preprocessing as bpp
import recommender.model_selection as rms
import recommender.preprocessing as rpp
import tag_genome_builder as tg_builder
import config
import config_tag_recommender
from recommender.base import ItemFeature

str_aggregated_path = config_tag_recommender.str_aggregated_path
str_movielens_movies_path = config_tag_recommender.str_movielense_movies_path
str_genome_scores = config_tag_recommender.str_genome_scores
str_tags = config_tag_recommender.str_tags
str_tag_ids = config_tag_recommender.str_tag_ids
str_rating_path = config_tag_recommender.str_rating_path
str_data_folder = config_tag_recommender.str_data_folder
minimum_no_of_frames = config_tag_recommender.minimum_no_of_frames


def get_df_vf() -> pd.DataFrame:
    usecols = ['movieId', 'no_key_frames']
    for i in range(1, 11):
        usecols += [f'f{i}_median', f'f{i}_quartile1', f'f{i}_quartile3', f'f{i}_std']
    print(len(usecols))
    df_agg = pd.read_csv(str_aggregated_path,
                         nrows=None,
                         usecols=usecols,
                         index_col=config.movieId_col).sort_index()
    print(df_agg.shape)
    df_agg = df_agg[df_agg['no_key_frames'] >= minimum_no_of_frames]
    df_agg.dropna(axis=1, thresh=len(df_agg) - 1000, inplace=True)
    assert not (df_agg.isnull().sum() > 0).any()
    return df_agg


def get_df_tags():
    list_of_movies = get_list_of_movies()
    df_tags = pd.read_csv(str_tags, nrows=None)
    df_tags.drop_duplicates(subset=[config.movieId_col, 'tag'], inplace=True)
    df_tags = df_tags[df_tags[config.movieId_col].isin(list_of_movies)].copy()
    return df_tags


def get_item_features_n_tags_from_df_tags(df_tags, number_of_tag_per_movie, random_state=7):
    df_tags_sampled = rpp.get_random_n_tags(df_tags, number_of_tag_per_movie,
                                            one_row_per_movie=True,
                                            random_state=random_state)
    count = CountVectorizer()
    feature_matrix = count.fit_transform(df_tags_sampled[config.tag_col].astype(str))
    item_ids = df_tags_sampled.index
    feature_names = count.get_feature_names()
    return ItemFeature(feature_matrix=feature_matrix,
                       feature_names=feature_names,
                       item_ids=item_ids)


def get_item_features_n_tags_from_tag_genome(df_genome, number_of_tag_per_movie):
    count = CountVectorizer()
    df_top_n_tags = tg_builder.Base().get_top_n_tags(df_genome,
                                                     n=number_of_tag_per_movie,
                                                     one_row_per_movie=True)
    feature_matrix = count.fit_transform(df_top_n_tags[config.tagId_col].astype(str))
    item_ids = df_top_n_tags.index
    feature_names = count.get_feature_names()
    return ItemFeature(feature_names=feature_names,
                       item_ids=item_ids,
                       feature_matrix=feature_matrix)


def get_list_of_movies():
    df_agg = get_df_vf()
    # df_ratings = pd.read_csv(str_rating_path, usecols=[config.movieId_col])
    # return df_agg.loc[df_agg[config.movieId_col].isin(df_ratings[config.movieId_col].unique()),
    #                   config.movieId_col]
    return df_agg.index.values


def normalize_vf(item_feature_train, item_feature_test, normalizer=pp.StandardScaler(with_mean=False)):
    if_normalizer = bpp.ItemFeatureNormalizer(normalizer=normalizer, inplace=True)
    item_feature_train_normalized = if_normalizer.fit_transform(item_feature_train)
    item_feature_test_normalized = if_normalizer.transform(item_feature_test)
    return item_feature_train_normalized, item_feature_test_normalized


def join_rating_to_item_features(df_rating: pd.DataFrame, item_features: ItemFeature, user_ids):
    assert set(df_rating[config.movieId_col]).issubset(item_features.item_ids) , 'sth'  # FIXME
    items = df_rating[config.movieId_col].values
    feature_for_items = item_features.get_item_feature_by_list_of_items(items).feature_matrix
    encoder = pp.OneHotEncoder(sparse=True, categories=[user_ids])
    user_ids_encoded = encoder.fit_transform(df_rating[[config.userId_col]])
    X = sparse.hstack([feature_for_items, user_ids_encoded])
    y = df_rating[config.rating_col]
    return X, y


def _filter_df_rating_item_features_to_same(df_rating_train: pd.DataFrame,
                                            df_rating_test: pd.DataFrame,
                                            item_features: ItemFeature):
    df_rating_train = df_rating_train[df_rating_train[config.movieId_col].isin(item_features.item_ids)]
    df_rating_test = df_rating_test[df_rating_test[config.movieId_col].isin(item_features.item_ids)]
    df_rating_train.sort_index(inplace=True)
    df_rating_test.sort_index(inplace=True)
    # item_features = item_features.get_item_feature_by_list_of_items(
    #     df_rating_train[config.movieId_col].unique())

    user_ids_train = sorted(df_rating_train[config.userId_col].unique())
    user_ids_test = df_rating_test[config.userId_col].unique()
    user_ids = sorted(list(set(user_ids_train).union(set(user_ids_test))))
    return df_rating_train, df_rating_test, item_features, user_ids


def get_ml_train_test(item_features: ItemFeature,
                      df_rating_train: pd.DataFrame,
                      df_rating_test: pd.DataFrame,
                      normalize=True) -> (sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray):
    df_rating_train, df_rating_test, item_features, user_ids =\
        _filter_df_rating_item_features_to_same(df_rating_train, df_rating_test, item_features)

    item_features_train = item_features.get_item_feature_by_list_of_items(
        df_rating_train[config.movieId_col].unique())
    item_features_test = item_features.get_item_feature_by_list_of_items(
        df_rating_test[config.movieId_col].unique())

    if normalize:
        item_features_train_normalized, item_features_test_normalized =\
            normalize_vf(item_features_train, item_features_test)
    else:
        item_features_train_normalized, item_features_test_normalized =\
            item_features_train, item_features_test

    X_train, y_train = join_rating_to_item_features(df_rating_train, item_features_train_normalized, user_ids)
    X_test, y_test = join_rating_to_item_features(df_rating_test, item_features_test_normalized, user_ids)
    assert np.allclose(df_rating_test[config.rating_col].values, y_test.values, .01)
    return X_train, y_train, X_test, y_test


def get_ml_train_test_vf(df_rating_train, df_rating_test):
    df_vf = get_df_vf()

    item_feature_vf = ItemFeature()
    item_feature_vf.from_dataframe(df_vf)
    return get_ml_train_test(item_feature_vf,
                             df_rating_train,
                             df_rating_test)


def get_ml_train_test_n_tags_df_tags(df_tags: pd.DataFrame, number_of_tag_per_movie: int,
                                     df_rating_train: pd.DataFrame, df_rating_test: pd.DataFrame,
                                     random_state: int = 7):
    item_feature_n_tags = get_item_features_n_tags_from_df_tags(df_tags, number_of_tag_per_movie,
                                                                random_state=random_state)
    return get_ml_train_test(item_feature_n_tags,
                             df_rating_train,
                             df_rating_test)


def get_ml_train_test_n_tags_df_genome(df_genome: pd.DataFrame, number_of_tag_per_movie: int,
                                       df_rating_train: pd.DataFrame, df_rating_test: pd.DataFrame,
                                       ):
    df_n_tags = get_item_features_n_tags_from_tag_genome(df_genome=df_genome,
                                                         number_of_tag_per_movie=number_of_tag_per_movie)
    return get_ml_train_test(df_n_tags,
                             df_rating_train,
                             df_rating_test)


def load_filter_and_split_df_rating(nrows):
    df_ratings = pd.read_csv(str_rating_path, nrows=nrows)
    list_of_movies = get_list_of_movies()
    df_ratings_filtered = df_ratings[df_ratings[config.movieId_col].isin(list_of_movies)]
    user_activities = df_ratings_filtered[config.userId_col].value_counts()
    df_ratings_filtered = df_ratings_filtered[df_ratings_filtered[config.userId_col].
                                              isin(user_activities[user_activities > 1].index)]
    # Train test split
    df_rating_train, df_rating_test = \
        rms.train_test_split(df_ratings_filtered, item_features=None, strategy='sklearn', test_size=.25)
    return df_rating_train, df_rating_test


def try_one(estimator,
            X_train: sparse.csr_matrix, y_train: np.ndarray,
            X_test: sparse.csr_matrix, y_test: np.ndarray,
            prediction_column_suffix) -> (str, np.ndarray):
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    print(f'RMSE {prediction_column_suffix}', np.sqrt(metrics.mean_squared_error(y_test, pred)))
    df = pd.DataFrame({prediction_column_suffix: pred})
    df.to_csv(os.path.join('.', 'output', f'{prediction_column_suffix}.csv'), index=False)
    return prediction_column_suffix, pred


def try_all(estimator, nrows=None, df_tags=None, df_tag_genome=None, df_tag_vf=None, n_jobs=5,
            n_tag_list=np.arange(1, 11), random_state=7):
    l_to_try = []
    df_rating_train, df_rating_test = load_filter_and_split_df_rating(nrows=nrows)
    print('vf')
    X_train_vf, y_train_vf, X_test_vf, y_test_vf =\
        get_ml_train_test_vf(df_rating_train=df_rating_train, df_rating_test=df_rating_test)
    l_to_try.append((clone(estimator), X_train_vf, y_train_vf, X_test_vf, y_test_vf, 'vf'))
    if df_tags is not None:
        for n_tags in n_tag_list:
            print(f'{n_tags} tag')
            X_train_tg, y_train_tg, X_test_tg, y_test_tg =\
                get_ml_train_test_n_tags_df_tags(df_tags=df_tags,
                                                 number_of_tag_per_movie=n_tags,
                                                 df_rating_train=df_rating_train,
                                                 df_rating_test=df_rating_test,
                                                 random_state=random_state)
            l_to_try.append((clone(estimator), X_train_tg, y_train_tg, X_test_tg, y_test_tg, f'tg_{n_tags}'))
    if df_tag_vf is not None:
        for n_tags in n_tag_list:
            print(f'{n_tags} vf_tg')
            X_train_tg, y_train_tg, X_test_tg, y_test_tg =\
                get_ml_train_test_n_tags_df_genome(df_genome=df_tag_vf,
                                                   number_of_tag_per_movie=n_tags,
                                                   df_rating_train=df_rating_train,
                                                   df_rating_test=df_rating_test)
            l_to_try.append((clone(estimator), X_train_tg, y_train_tg, X_test_tg, y_test_tg, f'vf_tg_{n_tags}'))

    recs = Parallel(n_jobs=n_jobs, verbose=30)(
        delayed(try_one)(
            p[0], p[1], p[2], p[3], p[4], p[5]
        )
        for p in l_to_try
    )
    for prediction_column_suffix, pred in recs:
        df_rating_test[f'{config.rating_col}_predicted_{prediction_column_suffix}'] = pred
        # df_rating_test.loc[null_indices,
        #                    f'{config.rating_col}_predicted_{prediction_column_suffix}'] = np.nan

    return df_rating_test
