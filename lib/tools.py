import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import time
from IPython.display import clear_output

import config


def plot_confusion_matrix(y_true, y_pred, classes=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          figsize=(5, 5)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if classes is not None:
        classes = classes[unique_labels(y_true, y_pred)]
    else:
        classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

#     print(cm)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def to_time(t0):
    t = time.time()
    t = t - t0
    h = int(t // 3600)
    m = int(t // 60 - h * 60)
    s = int(t - m * 60 - h * 3600)
    return f"{h}:{m}:{s}"


def print_status(total, n, t0=None):
    done = '#' * n
    todo = '-' * (total - n)
    s = '<{0}>'.format(done + todo) + f" {n}/{total}"
    if t0 is None:
        print(s, end='\r')
    else:
        print(s, 'Elapsed time:', to_time(t0), end='\r', flush=True)
    return n + 1


def update_progress(progress, t0=None):
    bar_length = 50
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait=True)
    if t0 is None:
        text = "Progress: [{0}] {1:.1f}%".format("#" * block + "-" * (bar_length - block),
                                                 progress * 100)
    else:
        text = "Progress: [{0}] {1:.1f}%, Elapsed time: {2:s}".format(
            "#" * block + "-" * (bar_length - block), progress * 100, to_time(t0))
    print(text)


def drop_nulls(df):
    subset = df.filter(regex='_predicted').columns
    return df.dropna(subset=subset)


def performance_report(df_rating_pred, prediction_column_suffix=''):
    if prediction_column_suffix != '':
        prediction_column_suffix = '_' + prediction_column_suffix
    mae = metrics.regression.mean_absolute_error(drop_nulls(df_rating_pred)[config.rating_col],
                                                 drop_nulls(df_rating_pred)
                                                 [f'{config.rating_col}_predicted{prediction_column_suffix}'])
    mse = metrics.regression.mean_squared_error(drop_nulls(df_rating_pred)[config.rating_col],
                                                drop_nulls(df_rating_pred)
                                                [f'{config.rating_col}_predicted{prediction_column_suffix}'])
    rmse = np.sqrt(mse)
    r2 = metrics.regression.r2_score(drop_nulls(df_rating_pred)[config.rating_col],
                                     drop_nulls(df_rating_pred)
                                     [f'{config.rating_col}_predicted{prediction_column_suffix}'])
    mean = df_rating_pred[config.rating_col].mean()
    nrmse = rmse / mean
    residual_std = df_rating_pred['residual' + prediction_column_suffix].std()
    residual_mean = df_rating_pred['residual' + prediction_column_suffix].mean()

    df_regression_report = pd.DataFrame({'Average Score': mean,
                                         'MAE': mae,
                                         'RMSE': rmse,
                                         'NRMSE': nrmse,
                                         'R2': r2,
                                         'Std of residuals': residual_std,
                                         'Avg of residuals': residual_mean
                                         },
                                        index=[prediction_column_suffix[1:]])
    return df_regression_report


def plot_prediction_histogram(df_rating_pred, ax=None, prediction_column_suffix='',
                              figsize=(7, 7), title=None, **kwargs):
    if title is None:
        title = 'Prediction distribution ' + prediction_column_suffix
    if prediction_column_suffix != '':
        prediction_column_suffix = '_' + prediction_column_suffix
    fig, ax = _make_fig(ax, figsize)
    df_rating_pred[[f'{config.rating_col}_predicted{prediction_column_suffix}']].plot.hist(ax=ax, **kwargs)
    ax.plot([0, 5], [0, 5], color='g')
    ax.set_title(title)
    return ax


def plot_actual_vs_prediction(df_rating_pred, ax=None, prediction_column_suffix='',
                              figsize=(7, 7), title=None, **kwargs):
    if title is None:
        title = 'Actual vs. predicted rating ' + prediction_column_suffix
    if prediction_column_suffix != '':
        prediction_column_suffix = '_' + prediction_column_suffix
    fig, ax = _make_fig(ax, figsize)
    df_rating_pred.plot.scatter(x=config.rating_col,
                                y=f'{config.rating_col}_predicted{prediction_column_suffix}',
                                ax=ax, **kwargs)
    ax.plot([0, 5], [0, 5], color='g')
    ax.set_title(title)
    return ax


def plot_residual_plot(df_rating_pred, ax=None, prediction_column_suffix='',
                       figsize=(7, 7), title=None, **kwargs):
    if title is None:
        title = 'Residual plot ' + prediction_column_suffix
    if prediction_column_suffix != '':
        prediction_column_suffix = '_' + prediction_column_suffix
    fig, ax = _make_fig(ax, figsize)
    df_rating_pred.plot.scatter(x=config.rating_col,
                                y=f'residual{prediction_column_suffix}',
                                ax=ax, **kwargs)
    ax.axhline(0, color='g')
    ax.set_title(title)
    return ax


def plot_residual_boxplot(df_rating_pred, ax=None, prediction_column_suffix='',
                          figsize=(7, 7), title=None, **kwargs):
    if title is None:
        title = 'Distribution of residuals per rating ' + prediction_column_suffix
    if prediction_column_suffix != '':
        prediction_column_suffix = '_' + prediction_column_suffix
    fig, ax = _make_fig(ax, figsize)
    sns.boxplot(x=config.rating_col, y=f'residual{prediction_column_suffix}',
                data=df_rating_pred, ax=ax, **kwargs)
    ax.set_title(title)
    return ax


def plot_absolute_residual_boxplot(df_rating_pred, ax=None, prediction_column_suffix='',
                                   figsize=(7, 7), title=None, **kwargs):
    if title is None:
        title = 'Distribution of absolute residuals per rating ' + prediction_column_suffix
    if prediction_column_suffix != '':
        prediction_column_suffix = '_' + prediction_column_suffix
    fig, ax = _make_fig(ax, figsize)
    sns.boxplot(x=config.rating_col, y=f'absolute residual{prediction_column_suffix}',
                data=df_rating_pred, ax=ax, **kwargs)
    ax.set_title(title)
    return ax


def plot_actual_vs_predicted_boxplot(df_rating_pred, ax=None, prediction_column_suffix='',
                                     figsize=(7, 7), title=None, **kwargs):
    if title is None:
        title = 'Distribution of predictions per rating ' + prediction_column_suffix
    if prediction_column_suffix != '':
        prediction_column_suffix = '_' + prediction_column_suffix
    fig, ax = _make_fig(ax, figsize)
    sns.boxplot(x=config.rating_col, y=f'{config.rating_col}_predicted{prediction_column_suffix}',
                data=df_rating_pred, ax=ax, **kwargs)
    ax.set_title(title)
    return ax


def plot_actual_vs_predicted_violinplot(df_rating_pred, ax=None, prediction_column_suffix='',
                                        figsize=(7, 7), title=None, **kwargs):
    if title is None:
        title = 'Distribution of predictions per rating ' + prediction_column_suffix
    if prediction_column_suffix != '':
        prediction_column_suffix = '_' + prediction_column_suffix
    fig, ax = _make_fig(ax, figsize)
    sns.violinplot(x=config.rating_col, y=f'{config.rating_col}_predicted{prediction_column_suffix}',
                   data=df_rating_pred, ax=ax, **kwargs)
    ax.set_title(title)
    return ax


def _make_fig(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    else:
        return None, ax


def plot_side_by_side(plotter, df_rating_pred, prediction_column_suffixes: list,
                      figsize=None, title='', **kwargs):
    n_rows = len(prediction_column_suffixes) // 2 + (len(prediction_column_suffixes) % 2)
    if figsize is None:
        figsize = (14, 5 * n_rows)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, sharey='all')
    for prediction_column_suffix, ax in zip(prediction_column_suffixes, axes.ravel()):
        plotter(df_rating_pred,
                prediction_column_suffix=prediction_column_suffix,
                ax=ax,
                **kwargs)
    fig.tight_layout()
    fig.suptitle(title)
    return fig

# from multiprocessing import Pool
# from tqdm import tqdm
# import functools
#
#
#
# def do_work(pbar, x):
#     # do something with x
#     time.sleep(.1)
#     pbar.update(1)
#     return x
#
#
# def m():
#     tasks = range(5)
#     pbar = tqdm(total=len(tasks))
#     pool = Pool()
#     _do_work = functools.partial(do_work, pbar)
#     a = pool.map(_do_work, tasks)
#     pool.close()
#     b = pool.join()
#     pbar.close()
#     return a, b
