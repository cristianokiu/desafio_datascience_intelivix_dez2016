import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence

# Obs: código adaptado de:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(cm, classes,
                          percentage=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys,
                          subplot=None):

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

    if percentage:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        subplot.text(j, i,
                ("{0:.2%}" if percentage else "{0}").format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    subplot.set_ylabel('True condition')
    subplot.set_xlabel('Predicted condition')


def plot_confusion_matrices(cms, classes,
                            percentage=False,
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
        plot_confusion_matrix(cm, classes, title=title,
                subplot=subplot, percentage=percentage)

    fig.set_size_inches(n_columns*4, n_rows*4)
    fig.tight_layout(pad=4)


class ConfusionMatrix(Sequence):
    def __init__(self, matrix, classes=None):
        super(Sequence, self).__init__()
        self.matrix = matrix
        if classes is None:
            classes = [None] * len(matrix)
        self.classes = [_ConfusionMatrixClass(self, i, c)
                for i, c in enumerate(classes)] 

    def __getitem__(self, i):
        return self.classes[i]
    def __len__(self):
        return len(self.classes)

class _ConfusionMatrixClass():
    def __init__(self, cm, i, name):
        self.cm = cm
        self.name = name
        self.i = i

        m = cm.matrix
        row_sum = sum(m[i,:])
        col_sum = sum(m[:,i])
        all_sum = sum(m.ravel())

        self.TP = m[i,i]
        self.FN = row_sum - self.TP
        self.FP = col_sum - self.TP
        self.TN = all_sum - (row_sum - col_sum)

    def __str__(self):
        return "{0} {1}".format(self.i, self.name or 'class')
