import itertools
import numpy as np
import matplotlib.pyplot as plt

# Obs: código adaptado de:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys,
                          subplot=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not subplot:
        subplot = plt.subplot()

    subplot.imshow(cm, interpolation='nearest', cmap=cmap)
    subplot.set_title(title)
    tick_marks = np.arange(len(classes))

    subplot.set_xticks(tick_marks)
    subplot.set_xticklabels(classes)
    for tick in subplot.get_xticklabels():
        tick.set_rotation(45)

    subplot.set_yticks(tick_marks)
    subplot.set_yticklabels(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        subplot.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    subplot.set_ylabel('True label')
    subplot.set_xlabel('Predicted label')


def plot_confusion_matrices(cms, classes,
                            normalize=False,
                            titles=None,
                            n_columns=3):
    n_cms = len(cms)
    if n_cms < n_columns:
        n_columns = n_cms

    if not titles:
        titles = ['Confusion matrix'] * n_cms

    n_rows = n_cms // n_columns + min(1, n_cms % n_columns)
    fig, subplots = plt.subplots(n_rows, n_columns)

    if n_rows > 1 and n_columns > 1:
        subplots = itertools.chain.from_iterable(subplots)
    elif n_rows == 1 and n_columns == 1:
        subplots = [subplots]

    for title, cm, subplot in zip(titles, cms, subplots):
        plot_confusion_matrix(cm, classes, title=title, subplot=subplot)

    fig.set_size_inches(n_columns*4, n_rows*4)
    fig.tight_layout(w_pad=4)
