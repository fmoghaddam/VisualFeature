import pandas as pd
import sklearn.preprocessing as pp
from scipy import sparse
from sklearn.base import clone
from lib import check_is_fitted
import config
movieId_col = config.movieId_col
tagId_col = config.tagId_col
relevance_col = config.relevance_col
visual_feature_col = config.visual_feature_col


class VisualFeatureToSparse(object):
    def __init__(self):
        self.encoder_visual_feature = pp.LabelEncoder()

    def _converter(self, df_agg_normalized: pd.DataFrame, encoder_movieid: pp.LabelEncoder, fit=True) \
            -> sparse.csr_matrix:
        """the encoder_movieid has to be fitted"""

        check_is_fitted(encoder_movieid, 'classes_')
        assert df_agg_normalized.index.name == movieId_col

        df_agg_melted = pd.melt(df_agg_normalized.reset_index(), id_vars=[movieId_col])
        if fit:
            self.encoder_visual_feature.fit(df_agg_melted['variable'])

        row_agg = self.encoder_visual_feature.transform(df_agg_melted['variable'])
        column_agg = encoder_movieid.transform(df_agg_melted[movieId_col])
        data_agg = df_agg_melted['value']
        coo_agg = sparse.coo_matrix((data_agg, (row_agg, column_agg)))
        csr_agg = coo_agg.tocsr()
        return csr_agg

    def fit(self, df_agg_normalized: pd.DataFrame, encoder_movieid: pp.LabelEncoder):
        self.fitted = True
        return self._converter(df_agg_normalized, encoder_movieid, True)

    def transform(self, df_agg_normalized: pd.DataFrame, encoder_movieid: pp.LabelEncoder):
        return self._converter(df_agg_normalized, encoder_movieid, False)


class VisualFeatureNormalizer(object):
    def __init__(self):
        # self.normalizer = clone(normalizer)
        pass

    def fit(self, df_vf, normalizer):
        self.normalizer = normalizer
        self.normalizer.fit(df_vf)

    def transform(self, df_vf: pd.DataFrame) -> (pd.DataFrame, object):
        check_is_fitted(self, 'normalizer')
        normalized_agg_visual_features = self.normalizer.transform(df_vf)
        df_agg_normalized = pd.DataFrame(normalized_agg_visual_features,
                                         columns=df_vf.columns, index=df_vf.index)
        return df_agg_normalized

    def fit_transform(self, df_vf: pd.DataFrame, normalizer: object) -> pd.DataFrame:
        self.fit(df_vf, normalizer=normalizer)
        return self.transform(df_vf)


visual_feature_to_sparse = VisualFeatureToSparse()
visual_feature_normalizer = VisualFeatureNormalizer()


class Base(object):
    @staticmethod
    def tag_genome_to_sparse(df_genome_scores: pd.DataFrame,
                             encoder_movieid: pp.LabelEncoder, encoder_tagid: pp.LabelEncoder)\
            -> sparse.csr_matrix:
        """compute the csr sparse matrix of tag genomes having *fitted* encoders"""
        check_is_fitted(encoder_movieid, 'classes_')
        check_is_fitted(encoder_tagid, 'classes_')
        rows = encoder_movieid.transform(df_genome_scores.movieId)
        columns = encoder_tagid.fit_transform(df_genome_scores.tagId)
        sparse_gnome = sparse.coo_matrix((df_genome_scores.relevance, (rows, columns)))
        csr_genome = sparse_gnome.tocsr()
        return csr_genome

    @staticmethod
    def normalize_df_agg_vf_transform(normalizer: object, df_vf: pd.DataFrame) -> (pd.DataFrame, object):
        _normalizer = clone(normalizer)
        normalized_agg_visual_features = _normalizer.fit_transform(df_vf)
        df_agg_normalized = pd.DataFrame(normalized_agg_visual_features,
                                         columns=df_vf.columns, index=df_vf.index)
        return df_agg_normalized, _normalizer

    @staticmethod
    def filter_vf_to_tag(df_agg, df_genome_scores):
        df_agg = df_agg[df_agg.index.isin(df_genome_scores.movieId)]
        return df_agg

    @staticmethod
    def filter_tag_to_vf(df_agg, df_genome_scores):
        df_genome_scores = df_genome_scores[df_genome_scores.movieId.isin(df_agg.index)]
        return df_genome_scores

    def filter_tag_and_vf_to_same(self, df_agg, df_genome_scores):
        df_genome_scores = self.filter_tag_to_vf(df_agg, df_genome_scores)
        df_agg = self.filter_vf_to_tag(df_agg, df_genome_scores)
        return df_agg, df_genome_scores

    @staticmethod
    def get_top_n_tags(df_tag_genome: pd.DataFrame, n: int = 10,
                       one_row_per_movie=False, sep='|') -> pd.DataFrame:
        df_tag_genome.sort_values(relevance_col, ascending=False, inplace=True)

        df_top_n_tags = df_tag_genome.groupby([movieId_col]).head(n).sort_values([movieId_col, tagId_col],
                                                                                 ascending=False)
        if one_row_per_movie:
            df_top_n_tags =\
                df_top_n_tags.groupby(config.movieId_col)[config.tagId_col].\
                agg(lambda x: sep.join(x.astype(str)))
        return pd.DataFrame(df_top_n_tags)


class TagGenomeBuilder(Base):
    def __init__(self, normalizer_vf, df_agg, df_genome_scores):
        if df_agg.isnull().sum().sum() > 0:
            print('df_agg has missing values, need to take care of them before fitting')
        if df_genome_scores.isnull().sum().sum() > 0:
            print('df_genome_scores has missing values, need to take care of them before fitting')
        assert df_agg.index.name == movieId_col
        self.encoder_movieid = pp.LabelEncoder()
        self.encoder_movieid.fit(df_agg.index)
        self.encoder_tagid = pp.LabelEncoder()
        self.encoder_tagid.fit(df_genome_scores.tagId)
        self.normalizer_vf = clone(normalizer_vf)

    def fit(self, df_genome_score: pd.DataFrame, df_visual_feature: pd.DataFrame):
        """
        :param df_genome_score: can be the total dataset, will be filtered according to df_visual_feature
        :param df_visual_feature: only included train data (movies has to be splitted)
        :return  sparse matrix of tag genome
        """
        assert df_visual_feature.isnull().sum().sum() == 0, ('df_visual_feature has missing values. Impute or'
                                                             ' remove them before fitting')
        assert df_genome_score.isnull().sum().sum() == 0, ('df_genome_score has missing values. Impute or'
                                                           ' remove them before fitting')
        self.l_visual_feature_names = df_visual_feature.columns.tolist()
        df_visual_feature, df_genome_score =\
            self.filter_tag_and_vf_to_same(df_agg=df_visual_feature, df_genome_scores=df_genome_score)

        df_agg_normalized =\
            visual_feature_normalizer.fit_transform(df_vf=df_visual_feature,
                                                    normalizer=self.normalizer_vf,
                                                    )
        csr_agg = visual_feature_to_sparse.fit(df_agg_normalized,
                                               encoder_movieid=self.encoder_movieid)
        csr_genome = self.tag_genome_to_sparse(df_genome_scores=df_genome_score,
                                               encoder_movieid=self.encoder_movieid,
                                               encoder_tagid=self.encoder_tagid)
        assert csr_agg.shape[1] == csr_genome.shape[0], (f'shapes of matrices are not compatible for dot '
                                                         f'product {csr_agg.shape} and {csr_genome.shape}')
        self.genome_score_visual_features = csr_agg.dot(csr_genome)
        return self.genome_score_visual_features

    def output_vf_genome_matrix_to_df(self, path_to_write: str = None) -> pd.DataFrame:
        """
        Convert the computed visual_feture genome matrix to a dataframe like the coo matrix for writing
        :param path_to_write: optional path to wrtite a datafraem to disk as .csv file
        """
        check_is_fitted(self, 'genome_score_visual_features')
        genome_score_visual_features_coo = self.genome_score_visual_features.tocoo()
        df_genome_visual_feature = pd.DataFrame(
            {
                visual_feature_col:
                    visual_feature_to_sparse.encoder_visual_feature.inverse_transform
                    (genome_score_visual_features_coo.row),
                tagId_col: self.encoder_tagid.inverse_transform(genome_score_visual_features_coo.col),
                relevance_col: genome_score_visual_features_coo.data
            }
        )
        df_genome_visual_feature.dropna(inplace=True)
        if path_to_write is not None:
            df_genome_visual_feature.to_csv(path_to_write)
        return df_genome_visual_feature

    def predict(self, df_visual_feature: pd.DataFrame, output_df: bool = False):
        """
        Compute the relevance to tags using visual features of new movies and the matrix computed in fit
        :param df_visual_feature: dataframe of aggregated visual features with movieId as index
        :param output_df: boolean if the matrix has to be converted to dataframe like
        output_vf_genome_matrix_to_df
        """
        check_is_fitted(self, 'genome_score_visual_features')
        check_is_fitted(visual_feature_normalizer, 'normalizer')
        assert df_visual_feature.isnull().sum().sum() == 0, ('df_visual_feature has missing values. Impute or'
                                                             ' remove them before fitting')
        assert df_visual_feature.index.name == movieId_col
        assert df_visual_feature.columns.tolist() == self.l_visual_feature_names, ('different set of visual '
                                                                                   'features comparing fitted'
                                                                                   ' ones')
        encoder_movie_id_transform = pp.LabelEncoder()
        encoder_movie_id_transform.fit(df_visual_feature.index)
        df_visual_feature_norm = visual_feature_normalizer.transform(df_visual_feature)
        visual_feature_norm_sparse =\
            visual_feature_to_sparse.transform(df_visual_feature_norm,
                                               encoder_movie_id_transform)
        csr_tag_genome_vf = visual_feature_norm_sparse.transpose().dot(self.genome_score_visual_features)
        if output_df:
            return self.output_by_vf_computed_tag_genome_to_df(csr_tag_genome_vf, encoder_movie_id_transform)
        else:
            return csr_tag_genome_vf

    def output_by_vf_computed_tag_genome_to_df(self, csr_tag_genome_vf: sparse.csr_matrix,
                                               encoder_movie_id_transform: pp.LabelEncoder) -> pd.DataFrame:
        coo_csr_tag_genome_vf = csr_tag_genome_vf.tocoo()
        df_tag_genome_vf = pd.DataFrame(
            {
                movieId_col: encoder_movie_id_transform.inverse_transform(coo_csr_tag_genome_vf.row),
                tagId_col: self.encoder_tagid.inverse_transform(coo_csr_tag_genome_vf.col),
                relevance_col: coo_csr_tag_genome_vf.data,
            }
        )
        return df_tag_genome_vf
