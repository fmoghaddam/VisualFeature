from base.data_types import ItemFeature
from lib import check_is_fitted


# def item_feature_nomalizer(normalizer, item_features: ItemFeature):
class ItemFeatureNormalizer(object):
    def __init__(self, normalizer, inplace: bool = True):
        self.normalizer = normalizer
        self.inplace = inplace

    def fit(self, item_features: ItemFeature):
        self.normalizer.fit(item_features.feature_matrix)
        self.fitted = True

    def transform(self, item_features):
        check_is_fitted(self, 'fitted')
        if self.inplace:
            item_features.feature_matrix = self.normalizer.transform(item_features.feature_matrix)
            return  item_features
        else:
            new = item_features.copy()
            new.feature_matrix = self.normalizer.transform(new.feature_matrix)
            return new

    def fit_transform(self, item_features):
        self.fit(item_features)
        return self.transform(item_features)

