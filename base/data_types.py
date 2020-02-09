from scipy import sparse
import numpy as np
import pandas as pd
import config

class ItemFeature(object):
    def __init__(self, item_ids: list = None, feature_names=None,
                 feature_matrix: sparse.csr_matrix = None):
        self.reserved_words = {config.movieId_col, config.userId_col, config.rating_col}
        if item_ids is not None:
            self._initiate(item_ids, feature_names, feature_matrix)

    def __len__(self):
        return self.shape[0]

    def _rename_reserved(self):
        if not self.reserved_words.isdisjoint(set(self.item_ids)):
            self.item_ids = self.item_ids.astype(object)
            for a in self.reserved_words:
                np.place(self.item_ids, self.item_ids == a, 'a' + '_')

    def _initiate(self, item_ids: list, feature_names: list,
                  feature_matrix: sparse.csr_matrix):
        self._validate_input(item_ids, feature_names, feature_matrix)
        self.item_ids = np.array(item_ids)
        self.feature_names = np.array(feature_names)
        self._rename_reserved()
        self.feature_matrix = feature_matrix
        self.shape = (len(item_ids), len(feature_names))

    def _validate_input(self, item_ids: list, feature_names: list,
                        feature_matrix: sparse.csr_matrix):
        if not isinstance(feature_matrix, sparse.csr_matrix):
            raise TypeError(f'only sparse.csr_matrix can be accepted as feature matrix'
                            f'{type(feature_matrix)} is given')
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
        assert set(some_item_ids).issubset(self.item_ids), ('I do not have all the items you wanted'
                                                            f'{set(some_item_ids).difference(self.item_ids)}'
                                                            'are missing')
        df_items_ids = pd.DataFrame({'indices': range(len(self.item_ids))},
                                    index=self.item_ids)
        item_ids_indices = df_items_ids.loc[some_item_ids, 'indices'].values
        return self.feature_matrix[item_ids_indices, :]
        # item_ids_indices = np.array([np.where(self.item_ids == item)[0][0]
        #                              for item in some_item_ids
        #                              if item in self.item_ids])
        # return self.feature_matrix[item_ids_indices, :]

    def get_item_feature_by_list_of_items(self, some_item_ids):
        return ItemFeature(item_ids=some_item_ids,
                           feature_names=self.feature_names,
                           feature_matrix=self.get_feature_matrix_by_list_of_items(some_item_ids))

    def to_dataframe(self):
        df = pd.DataFrame(self.feature_matrix.toarray(),
                          index=self.item_ids,
                          columns=self.feature_names)
        df.index.name = config.movieId_col
        return df

    def copy(self):
        new = ItemFeature(feature_matrix=self.feature_matrix,
                          feature_names=self.feature_names,
                          item_ids=self.item_ids)
        return new
