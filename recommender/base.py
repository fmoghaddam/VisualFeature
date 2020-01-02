from scipy import sparse
import numpy as np
import pandas as pd


class ItemFeature(object):
    def __init__(self, item_ids: list = None, feature_names=None,
                 feature_matrix: sparse.csr_matrix = None):
        if item_ids is not None:
            self._initiate(item_ids, feature_names, feature_matrix)

    def __len__(self):
        return self.shape[0]

    def _initiate(self, item_ids: list, feature_names: list,
                  feature_matrix: sparse.csr_matrix):
        self._validate_input(item_ids, feature_names, feature_matrix)
        self.item_ids = np.array(item_ids)
        self.feature_names = np.array(feature_names)
        self.feature_matrix = feature_matrix
        self.shape = (len(item_ids), len(feature_names))

    def _validate_input(self, item_ids: list, feature_names: list,
                        feature_matrix: sparse.csr_matrix):
        if not isinstance(feature_matrix, sparse.csr_matrix):
            raise TypeError('only sparse.csr_matrix can be accepted as feature matrix')
        assert feature_matrix.shape == (len(item_ids), len(feature_names)), ('dimension mismatch, '
                                                                             'feature_matrix does not have '
                                                                             'compatible shape comparing to '
                                                                             'number of items and number of '
                                                                             'features')
        no_of_nulls_in_feature_matrix = pd.isnull(feature_matrix.data).sum()
        if no_of_nulls_in_feature_matrix > 0:
            raise ValueError(f'feature matrix contains {no_of_nulls_in_feature_matrix}'
                             f' missing values. Do something about them first')

    def from_dataframe(self, df: pd.DataFrame):
        """df has movieId's as index and feature names as columns"""
        self._initiate(df.index, df.columns, sparse.csr_matrix(df.values))

    def get_feature_matrix_by_list_of_items(self, some_item_ids):
        assert set(some_item_ids).issubset(self.item_ids), 'I do not have all the items you wanted'
        item_ids_indices = np.array([np.where(self.item_ids == item)[0][0]
                                     for item in some_item_ids
                                     if item in self.item_ids])
        return self.feature_matrix[item_ids_indices, :]

    def get_item_feature_by_list_of_items(self, some_item_ids):
        return ItemFeature(item_ids=some_item_ids,
                           feature_names=self.feature_names,
                           feature_matrix=self.get_feature_matrix_by_list_of_items(some_item_ids))
