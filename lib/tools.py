import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output


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
