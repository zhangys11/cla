import os
import sys
import numpy as np
import matplotlib.pyplot as plt

if __package__:
    from .unsupervised_dimension_reductions import unsupervised_dimension_reductions
else:
    VIS_DIR = os.path.dirname(__file__)
    if VIS_DIR not in sys.path:
        sys.path.append(VIS_DIR)
    from unsupervised_dimension_reductions import unsupervised_dimension_reductions

def plot_feature_importance(feature_importances, title, row_size=100, figsize=(12, 2)):
    '''
    plot the importance for all features
    '''
    feature_importances = np.array(feature_importances)

    # matrix chart
    ROW_SIZE = row_size  # math.ceil(feature_importances.size / 100) * 10
    rows = int((feature_importances.size - 1)/ROW_SIZE) + 1
    fig = plt.figure(figsize=(12, 12*rows/ROW_SIZE+0.5))
    if title:
        plt.title(title + "\n" + 'importance marked by color depth')
    plt.axis('off')

    for i in range(rows):
        ax = fig.add_subplot(rows, 1, i + 1)
        ax.axis('off')
        arr = feature_importances[i*ROW_SIZE: min(
            i*ROW_SIZE+ROW_SIZE - 1, feature_importances.size - 1)]
        s = ax.matshow(arr.reshape(1, -1), cmap=plt.cm.Blues)

    # bar chart
    plt.figure(figsize=figsize)
    if title:
        plt.title(title + "\n" + 'importance marked by bar height')
    plt.bar(range(feature_importances.size),
            feature_importances, alpha=1.0, width=10)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def visualize_important_features(X, y, coef, X_names=None, title=None, row_size=100, eps=1e-10):
    '''
    epsilon - cut threshold
    '''
    N = np.count_nonzero(np.abs(coef) > eps)
    # take last N item indices and reverse (ord desc)
    biggest_fs = np.argsort(np.abs(coef))[::-1][:N]
    X_fs = X[:, biggest_fs]  # 前N个系数 non-zero

    print('Important feature Number:', N)
    print('Important feature Indice:', biggest_fs)
    if (X_names and len(X_names) == X.shape[1]):
        print('Important features:', np.array(X_names)[biggest_fs])
    print('Important feature coefficents:', coef[biggest_fs])

    plot_feature_importance(np.abs(coef), title, row_size=row_size)
    unsupervised_dimension_reductions(X_fs, y)


def get_important_features(X, coef, eps=1e-10):

    N = np.count_nonzero(np.abs(coef) > eps)
    biggest_fs = np.argsort(np.abs(coef))[::-1][:N]
    return X[:, biggest_fs]
