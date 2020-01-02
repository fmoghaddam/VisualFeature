from sklearn.feature_extraction.text import CountVectorizer
import tag_genome_builder as tg_builder
from recommender.base import ItemFeature
import config


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
