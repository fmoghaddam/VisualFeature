from scipy import sparse


class ItemBasedColabSin(object):
    def __init__(self):
        pass

    def fit(self, df_rating, df_item_features):
        """make a dictionary {user_id, (list of rated movies, np.array of respective rates)"""
        pass

    def predict(self, user_id, new_items):
        """for the given user_id give the predicted rates for new_items"""
        pass

    def items_to_feature_space(self, df) -> sparse.csr_matrix:
        pass
