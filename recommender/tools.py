import numpy as np
import pandas as pd
import config


def prepare_recommendations_df(df_rating_test, recommendations, prediction_column_suffix):
    recommendations.rename(columns={f'{config.rating_col}_predicted':
                                    f'{config.rating_col}_predicted_{prediction_column_suffix}'},
                           inplace=True)

    if f'{config.rating_col}_predicted_{prediction_column_suffix}' in df_rating_test.columns:
        df_rating_test.drop(columns=[f'{config.rating_col}_predicted_{prediction_column_suffix}'],
                            inplace=True)
    df_rating_test = pd.merge(df_rating_test, recommendations, on=[config.userId_col, config.movieId_col])
    # print(df_rating_test.columns)
    df_rating_test[f'residual_{prediction_column_suffix}'] = df_rating_test[config.rating_col] - \
        df_rating_test[f'{config.rating_col}_predicted_{prediction_column_suffix}']
    df_rating_test[f'absolute residual_{prediction_column_suffix}'] = \
        df_rating_test[f'residual_{prediction_column_suffix}'].abs()

    no_of_null_predictions = df_rating_test[f'{config.rating_col}_predicted_{prediction_column_suffix}'].\
        isnull().sum()
    share_of_null_predictions = df_rating_test[f'{config.rating_col}_predicted_{prediction_column_suffix}'].\
        isnull().mean()

    if no_of_null_predictions > 0:
        print(f'Warning: {no_of_null_predictions} ({np.round(share_of_null_predictions * 100 , 2)}%)'
              ' of ratings has no prediction')
    return df_rating_test
