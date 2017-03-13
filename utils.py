import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence


class ConfusionMatrix(Sequence):
    def __init__(self, matrix, name=None, classes=None):
        self.matrix = np.array(matrix)
        self.name = name
        self.classes = classes

        n_classes = len(matrix)
        self.multiclass = n_classes > 2

        self._sub_matrices = []
        if self.multiclass:
            self.TP, self.FN, self.FP, self.TN = np.empty((4, n_classes))
            for i in range(n_classes):
                row_sum = sum(matrix[i,:])
                col_sum = sum(matrix[:,i])
                all_sum = sum(matrix.ravel())

                tp = matrix[i,i]
                fn = row_sum - tp
                fp = col_sum - tp
                tn = all_sum - (row_sum - col_sum)
                m = [[tp, fn], [fp, tn]]
                n = None if classes is None else classes[i]
                
                self.TP[i], self.FN[i] = tp, fn
                self.FP[i], self.TN[i] = fp, tn
                self._sub_matrices.append(ConfusionMatrix(m, n))
        else:
            self._sub_matrices = []
            self.TP, self.FN = matrix[0]
            self.FP, self.TN = matrix[1]

    def __getitem__(self, i):
        return self._sub_matrices[i]
    def __len__(self):
        return len(self._sub_matrices)

    def __str__(self):
        return str(self.matrix)


# Obs: os plots foram adaptados de:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(confusion_matrix,
                          percentage=False,
                          cmap=plt.cm.Greys,
                          subplot=None):

    cm = confusion_matrix.matrix
    classes = confusion_matrix.classes

    if classes is None:
        if confusion_matrix.multiclass:
            classes = [str(i) for i in range(confusion_matrix.matrix)]
        else:
            classes = ['True', 'False']

    title = confusion_matrix.name or 'Confusion matrix'

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


def plot_confusion_matrices(confusion_matrices,
                            percentage=False,
                            title=None,
                            n_columns=3):

    n_cms = len(confusion_matrices)
    if n_cms < n_columns:
        n_columns = n_cms
    
    if not title and hasattr(confusion_matrices, 'name'):
        title = confusion_matrices.name

    n_rows = n_cms // n_columns + min(1, n_cms % n_columns)
    fig, subplots = plt.subplots(n_rows, n_columns)

    if n_rows > 1 and n_columns > 1:
        subplots = itertools.chain.from_iterable(subplots)
    elif n_rows == 1 and n_columns == 1:
        subplots = [subplots]

    for confusion_matrix, subplot in zip(confusion_matrices, subplots):
        plot_confusion_matrix(confusion_matrix,
                subplot=subplot, percentage=percentage)


    if title:
        fig.suptitle(title, fontsize=16)
    fig.set_size_inches(n_columns*4, n_rows*4)
    fig.tight_layout(pad=4)
