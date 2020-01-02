import numpy as np
import pandas as pd
import config

df_top_n_sample = pd.DataFrame({config.tagId_col: '786|785|589|588|536|244|204|186|64|63'},
                               index=pd.Index([1], name=config.movieId_col))


def test_get_item_feature_from_tag_genome():
    # TODO mock get_top_n_tags to return df_top_n_sample
    expected_items_ids = np.array([1])
    expected_feature_names = np.array(['186', '204', '244', '536', '588', '589', '63', '64', '785', '786'])
    expected_feature_matrix_array = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
