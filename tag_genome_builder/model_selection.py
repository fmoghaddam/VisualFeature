import pandas as pd
import sklearn.model_selection as ms
from tag_genome_builder import TagGenomeBuilder
import functools
import multiprocessing
from tqdm import tqdm_notebook as tqdm


def loop(estimator, df_visual_features, df_genome_scores, splits):
    _movies_train_place, _movies_test_place = splits
    _movies_train = df_visual_features.index.values[_movies_train_place]
    _movies_test = df_visual_features.index.values[_movies_test_place]

    _df_vf_train = df_visual_features.loc[_movies_train]
    _df_vf_test = df_visual_features.loc[_movies_test]
    estimator.fit(df_genome_score=df_genome_scores, df_visual_feature=_df_vf_train)
    return estimator.predict(_df_vf_test, output_df=True)


def cross_val_predict(df_visual_features, df_genome_scores, normalizer_vf, n_splits, n_jobs=1):
    folds = ms.KFold(n_splits=n_splits, shuffle=True)
    tag_genome_builder = TagGenomeBuilder(df_agg=df_visual_features,
                                          df_genome_scores=df_genome_scores,
                                          normalizer_vf=normalizer_vf)
    l_tag_genome_vfs = []
    if n_jobs == 1:
        for splits in folds.split(df_visual_features.index):
            l_tag_genome_vfs.append(loop(tag_genome_builder, df_visual_features, df_genome_scores, splits))
            print('.', end=' ')
    else:
        _loop = functools.partial(loop, tag_genome_builder, df_visual_features, df_genome_scores)
        if n_jobs < 0:
            assert n_jobs > -multiprocessing.cpu_count(), 'too small negative n_jobs'
            n_jobs = multiprocessing.cpu_count() + n_jobs
        pool = multiprocessing.Pool(n_jobs)
        for tag_genome in tqdm(pool.imap(_loop, folds.split(df_visual_features.index)), total=n_splits):
            l_tag_genome_vfs.append(tag_genome)
        # l_tag_genome_vfs = pool.map(_loop, folds.split(df_visual_features.index))
        pool.close()
    return pd.concat(l_tag_genome_vfs, ignore_index=True)
