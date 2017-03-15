import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
import seaborn as sns


class ConfusionMatrix(Sequence):
    def __init__(self, matrix, name=None, classes=None):
        self.matrix = np.array(matrix)
        self.name = name
        self.classes = classes

        n_classes = len(matrix)
        self.multiclass = n_classes > 2

        self._sub_matrices = []
        if self.multiclass:
            self.TP = self.FN = self.FP = self.TN = 0
            for i in range(n_classes):
                row_sum = matrix[i,:].sum()
                col_sum = matrix[:,i].sum()
                all_sum = matrix.sum()

                tp = matrix[i,i]
                fn = row_sum - tp
                fp = col_sum - tp
                tn = all_sum - (row_sum - col_sum)
                m = [[tp, fn], [fp, tn]]
                n = None if classes is None else classes[i]

                self.TP += tp / n_classes
                self.FN += fn / n_classes
                self.FP += fp / n_classes
                self.TN += tn / n_classes
                
                self._sub_matrices.append(ConfusionMatrix(m, n))
        else:
            self._sub_matrices = []
            self.TP, self.FN = matrix[0]
            self.FP, self.TN = matrix[1]

        self.metrics = ConfusionMatrixMetrics(self)

    def __getitem__(self, i):
        return self._sub_matrices[i]
    def __len__(self):
        return len(self._sub_matrices)

    def __str__(self):
        return str(self.matrix)

class ConfusionMatrixMetrics(Sequence):
    def __init__(self, confusion_matrix):
        cm = confusion_matrix
        TP, FN = np.float64(cm.TP), np.float64(cm.FN)
        FP, TN = np.float64(cm.FP), np.float64(cm.TN)
        self.confusion_matrix = cm
        self.list = []

        self._new('TPR', TP / (TP + FN), 'Sensitivity, recall, hit rate, or true positive rate')
        self._new('TNR', TN / (FP + TN), 'Specificity or true negative rate')
        self._new('PPV', TP / (TP + FP), 'Precision or positive predictive value')
        self._new('NPV', TN / (TN + FN), 'Negative predictive value')
        self._new('FPR', 1 - self.TNR, 'Fall-out or false positive rate')
        self._new('FDR', 1 - self.PPV, 'False discovery rate')
        self._new('FNR', 1 - self.TPR, 'Miss rate or false negative rate')

        # self._new('ACC', (TP + TN) / (TP + FN + FP + TN), 'Accuracy')
        self._new('ACC', cm.matrix.diagonal().sum() / cm.matrix.sum(), 'Accuracy')
        self._new('F1', (2 * TP) / (2 * TP + FP + FN), 'F1 score - is the harmonic mean of precision and sensitivity')
        self._new('MCC', (TP * TN - FP * FN) / np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)), 'Matthews correlation coefficient')
        self._new('BM', self.TPR + self.TNR - 1, 'Informedness or Bookmaker Informedness')
        self._new('MK', self.PPV + self.NPV - 1, 'Markedness')

    def _new(self, acronym, value, description=None):
        self.__dict__[acronym] = value
        self.list.append((description, acronym, value))

    def __getitem__(self, i):
        return self.list[i]
    def __len__(self):
        return len(self.list)

    def __str__(self):
        return '\n\n'.join('{0}\n{1}: {2:.2%}'.format(d, a, v)
                for a, v, d in self)


def plot_confusion_matrix(confusion_matrix,
                          percentage=False,
                          cmap=plt.cm.PuBu,
                          subplot=None):

    cm = confusion_matrix.matrix
    classes = confusion_matrix.classes

    if classes is None:
        if confusion_matrix.multiclass:
            classes = [str(i) for i in range(confusion_matrix.matrix)]
        else:
            classes = ['True', 'False']

    title = confusion_matrix.name or 'Confusion matrix'

    if percentage:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if not subplot:
        subplot = plt.subplot()

    sns.heatmap(cm,
            xticklabels=classes,
            yticklabels=classes,
            annot=True,
            fmt= '.2%' if percentage else 'd',
            ax=subplot,
            cbar=False,
            cmap=cmap
            )

    subplot.set_title(title)
    subplot.set_ylabel('True condition')
    subplot.set_xlabel('Predicted condition')

    for tick in subplot.get_xticklabels():
        tick.set_rotation(45)
    for tick in subplot.get_yticklabels():
        tick.set_rotation(0)


def plot_confusion_matrices(confusion_matrices,
                            percentage=False,
                            title=None,
                            n_columns=3,
                            cmap=plt.cm.PuBu,
                            ):

    n_cms = len(confusion_matrices)
    if n_cms < n_columns:
        n_columns = n_cms
    
    if not title and hasattr(confusion_matrices, 'name'):
        title = confusion_matrices.name

    n_rows = n_cms // n_columns + min(1, n_cms % n_columns)

    fig, subplots = plt.subplots(n_rows, n_columns,
            figsize=(n_columns*4, n_rows*4.2)
            )

    if n_rows > 1 and n_columns > 1:
        subplots = itertools.chain.from_iterable(subplots)
    elif n_rows == 1 and n_columns == 1:
        subplots = [subplots]

    for confusion_matrix, subplot in zip(confusion_matrices, subplots):
        plot_confusion_matrix(confusion_matrix,
                subplot=subplot, percentage=percentage, cmap=cmap)


    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout(pad=4)

