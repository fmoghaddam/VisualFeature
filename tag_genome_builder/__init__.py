import pandas as pd
import sklearn.preprocessing as pp
from scipy import sparse
from sklearn.base import clone


class VisualFeatureToSparse(object):
    def __init__(self):
        self.encoder_visual_feature = pp.LabelEncoder()

    def _converter(self, df_agg_normalized: pd.DataFrame, encoder_movieid: pp.LabelEncoder, fit=True) \
            -> (sparse.csr_matrix, pp.LabelEncoder):
        """the encoder_movieid has to be fitted"""

        pp.label.check_is_fitted(encoder_movieid, 'classes_')

        df_agg_melted = pd.melt(df_agg_normalized.reset_index(), id_vars=['movieId'])
        if fit:
            self.encoder_visual_feature.fit(df_agg_melted['variable'])

        row_agg = self.encoder_visual_feature.transform(df_agg_melted['variable'])
        column_agg = encoder_movieid.transform(df_agg_melted['movieId'])
        data_agg = df_agg_melted['value']
        coo_agg = sparse.coo_matrix((data_agg, (row_agg, column_agg)))
        csr_agg = coo_agg.tocsr()
        return csr_agg

    def fit(self, df_agg_normalized: pd.DataFrame, encoder_movieid: pp.LabelEncoder):
        return self._converter(df_agg_normalized, encoder_movieid, True)

    def transform(self, df_agg_normalized: pd.DataFrame, encoder_movieid: pp.LabelEncoder):
        return self._converter(df_agg_normalized, encoder_movieid, False)


visual_feature_to_sparse = VisualFeatureToSparse()


class Base(object):
    @staticmethod
    def tag_genome_to_sparse(df_genome_scores: pd.DataFrame,
                             encoder_movieid: pp.LabelEncoder, encoder_tagid: pp.LabelEncoder)\
            -> sparse.csr_matrix:
        """compute the csr sparse matrix of tag genomes having *fitted* encoders"""
        pp.label.check_is_fitted(encoder_movieid, 'classes_')
        pp.label.check_is_fitted(encoder_tagid, 'classes_')
        rows = encoder_movieid.transform(df_genome_scores.movieId)
        columns = encoder_tagid.fit_transform(df_genome_scores.tagId)
        sparse_gnome = sparse.coo_matrix((df_genome_scores.relevance, (rows, columns)))
        csr_genome = sparse_gnome.tocsr()
        return csr_genome

    @staticmethod
    def normalize_df_agg_vf(normalizer: object, df_vf: pd.DataFrame) -> (pd.DataFrame, object):
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


class TagGenomeBuilder(Base):
    def __init__(self, normalizer_vf, df_agg, df_genome_scores):
        self.encoder_movieid = pp.LabelEncoder()
        self.encoder_movieid.fit(df_agg.index)
        self.encoder_tagid = pp.LabelEncoder()
        self.encoder_tagid.fit(df_genome_scores.tagId)
        # self.encoder_visual_feature = None  #pp.LabelEncoder()
        # encoder_visual_feature.fit(df_agg.columns)
        self.normalizer_vf = clone(normalizer_vf)

    def fit(self, df_genome_score: pd.DataFrame, df_visual_feature: pd.DataFrame):
        """
        :param df_genome_score: only included train data (movies has to be splitted)
        :param df_visual_feature: can be the total dataset, will be filtered according to
        df_genome_score

        :return  sparse matrix of tag genome
        """
        df_visual_feature, df_genome_score =\
            self.filter_tag_and_vf_to_same(df_agg=df_visual_feature, df_genome_scores=df_genome_score)

        df_agg_normalized, self.normalizer_vf =\
            self.normalize_df_agg_vf(normalizer=self.normalizer_vf,
                                     df_vf=df_visual_feature)
        csr_agg = visual_feature_to_sparse.fit(df_agg_normalized,
                                               encoder_movieid=self.encoder_movieid)
        csr_genome = self.tag_genome_to_sparse(df_genome_scores=df_genome_score,
                                               encoder_movieid=self.encoder_movieid,
                                               encoder_tagid=self.encoder_tagid)
        assert csr_agg.shape[1] == csr_genome.shape[0], (f'shapes od matrices are not compatible for dot'
                                                         f'product {csr_agg.shape} and {csr_genome.shape}')
        self.genome_score_visual_features = csr_agg.dot(csr_genome)

    def transform(self, df_visual_feature: pd.DataFrame):
        pp.label.check_is_fitted(self, 'genome_score_visual_features')
        pp.label.check_is_fitted(visual_feature_to_sparse,
                                 'encoder_visual_feature.classes_')
        df_visual_feature_norm = self.normalizer_vf.transform(df_visual_feature)
        df_visual_feature_norm_sparse = visual_feature_to_sparse.transform(df_visual_feature_norm,
                                                                           self.encoder_movieid)
        csr_tag_genome_vf = df_visual_feature_norm_sparse.dot(self.genome_score_visual_features)
        return csr_tag_genome_vf

    def output_vf_genome_matrix_to_df(self, path_to_write: str = None) -> pd.DataFrame:
        pp.label.check_is_fitted(self, 'genome_score_visual_features')
        genome_score_visual_features_coo = self.genome_score_visual_features.tocoo()
        df_genome_visual_feature = pd.DataFrame(
            {'visual_feature':
                visual_feature_to_sparse.encoder_visual_feature.inverse_transform
                (genome_score_visual_features_coo.row),
             'tagId': self.encoder_tagid.inverse_transform(genome_score_visual_features_coo.col),
             'relevance': genome_score_visual_features_coo.data})
        df_genome_visual_feature.dropna(inplace=True)
        if path_to_write is not None:
            df_genome_visual_feature.to_csv(path_to_write)
        return df_genome_visual_feature

