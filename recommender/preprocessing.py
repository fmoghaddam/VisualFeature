import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import tag_genome_builder as tg_builder
from lib import check_is_fitted
from recommender.base import ItemFeature
import config

movie_rating_cols = [config.movieId_col, config.userId_col, config.rating_col]


def get_item_feature_from_tag_genome(df_genome, number_of_tag_per_movie):
    count = CountVectorizer()
    df_top_n_tags = tg_builder.Base().get_top_n_tags(df_genome,
                                                     n=number_of_tag_per_movie,
                                                     one_row_per_movie=True)
    feature_matrix = count.fit_transform(df_top_n_tags[config.tagId_col].astype(str))
    item_ids = df_top_n_tags.index
    feature_names = count.get_feature_names()

    item_features = ItemFeature(item_ids=item_ids,
                                feature_names=feature_names,
                                feature_matrix=feature_matrix)
    return item_features


def get_random_n_tags(df_tags, number_of_tag_per_movie, one_row_per_movie, sep='|', random_state=None):
    df_tags_sampled = df_tags[[config.movieId_col, 'tag']].groupby(config.movieId_col, as_index=False).apply(
        lambda x:
        x.sample(number_of_tag_per_movie, random_state=random_state)
        if len(x) > number_of_tag_per_movie else x
    )
    if one_row_per_movie:
        df_tags_sampled = \
            df_tags_sampled.groupby(config.movieId_col)[config.tag_col]. \
            agg(lambda x: sep.join(x.astype(str)))
    return pd.DataFrame(df_tags_sampled)


def get_item_feature_from_tag(df_tags, number_of_tag_per_movie, random_state=None):
    count = CountVectorizer()
    df_tags_sampled = get_random_n_tags(df_tags, number_of_tag_per_movie,
                                        one_row_per_movie=True,
                                        random_state=random_state)
    feature_matrix = count.fit_transform(df_tags_sampled[config.tag_col].astype(str))
    item_ids = df_tags_sampled.index
    feature_names = count.get_feature_names()

    item_features = ItemFeature(item_ids=item_ids,
                                feature_names=feature_names,
                                feature_matrix=feature_matrix)
    return item_features


class RatingNormalizer(object):
    def __init__(self):
        pass

    def fit(self, df_rating: pd.DataFrame):
        self._validate_input(df_rating)
        self.df_fit = df_rating.groupby(config.userId_col)[[config.rating_col]].agg('mean')
        return self

    def transform(self, df_rating):
        check_is_fitted(self, 'df_fit')
        scaled_ratings = df_rating[config.rating_col].values -\
            self.df_fit.loc[df_rating[config.userId_col], config.rating_col].values
        return scaled_ratings

    def fit_transform(self, df_rating):
        self.fit(df_rating)
        return self.transform(df_rating)

    def inverse_transform(self, df_rating):
        self._validate_input(df_rating)
        descaled_ratings = df_rating[config.rating_col].values + \
            self.df_fit.loc[df_rating[config.userId_col], config.rating_col].values
        return descaled_ratings

    def _validate_input(self, df_rating):
        if not isinstance(df_rating, pd.DataFrame):
            raise TypeError('Only pandas DataFrame are accepted as input for rating')
        assert set(movie_rating_cols).issubset(df_rating.columns), ('df_rating has to have at least these '
                                                                    f'columns: {movie_rating_cols}')
