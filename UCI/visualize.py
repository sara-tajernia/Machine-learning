import numpy as np
from sklearn.manifold import TSNE

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def visualize_acc(info, accept, info_list):
    #not normalized
    info_list = info_list.transpose()
    for i in range(len(info_list)):
        list1, list2 = zip(*sorted(zip(info_list[i], accept)))
        plt.plot(list1, list2)
        plt.title(i, color='g', fontsize=25)
        plt.xlabel(i)
        plt.ylabel("acception")
        # plt.show()

    #nirmalized
    for i in info:
        list1, list2 = zip(*sorted(zip(info[i], accept)))
        plt.plot(list1, list2)
        plt.title(i, color='g', fontsize=25)
        plt.xlabel(i)
        plt.ylabel("acception")
        # plt.show()


def visualize_count(info):
    for i in info:
        dict_count = {x: info[i].count(x) for x in info[i]}
        list_n = dict(sorted(dict_count.items()))
        keys = list(list_n.keys())
        values = list(list_n.values())
        plt.plot(keys, values)
        plt.title(i, color='g', fontsize=25)
        plt.xlabel(i)
        plt.ylabel("count")
        # plt.show()


def visualize_3d(X, y_true, y_pre):
    info_TSNE = TSNE(n_components=2, learning_rate='auto', init = 'random')
    X = np.array(X)
    info_TSNE = info_TSNE.fit_transform(X)
    info_TSNE = info_TSNE.transpose()
    X = info_TSNE[0]
    Y = info_TSNE[1]
    colors = ['blue', 'red']
    cmap = mcolors.ListedColormap(colors)
    fig = plt.figure()
    plt.title('True')
    dims = y_true
    sc = plt.scatter(X, Y, c=dims, marker="o", cmap=cmap)
    fig.colorbar(sc)
    plt.show()

    colors = ['blue', 'red']
    cmap = mcolors.ListedColormap(colors)
    fig = plt.figure()
    plt.title('Prediction')
    dims = y_pre
    sc = plt.scatter(X, Y, c=dims, marker="o", cmap=cmap)
    fig.colorbar(sc)
    plt.show()

    dims = []
    for i in range(len(y_true)):
        if y_true[i] == y_pre[i] and y_true[i] == 0:
            dims.append(0)
        elif y_true[i] == y_pre[i] and y_true[i] == 1:
            dims.append(1)
        else:
            dims.append(2)
    # blue-> trur predic class 0     green-> trur predic class 1     red -> wrong predict
    colors = ['blue', 'green', 'red']
    cmap = mcolors.ListedColormap(colors)
    fig = plt.figure()
    plt.title('mistakes')
    sc = plt.scatter(X, Y, c=dims, marker="o", cmap=cmap)
    fig.colorbar(sc)
    plt.show()

def visualize_truth(X, y_true, y_pre):
    dims = []
    for i in range(len(y_true)):
        if y_true[i] == y_pre[i] and y_true[i] == 0:
            dims.append(0)
        elif y_true[i] == y_pre[i] and y_true[i] == 1:
            dims.append(1)
        else:
            dims.append(2)
    X = np.array(X)
    info_TSNE = TSNE(n_components=2, learning_rate='auto', init = 'random')
    info_TSNE = info_TSNE.fit_transform(X)
    info_TSNE = info_TSNE.transpose()
    X = info_TSNE[0]
    Y = info_TSNE[1]
    #blue-> trur predic class 0     green-> trur predic class 1     red -> wrong predict
    colors = ['blue', 'green', 'red']
    cmap = mcolors.ListedColormap(colors)
    fig = plt.figure()
    plt.title('True')
    sc = plt.scatter(X, Y, c=dims, marker="o", cmap=cmap)
    fig.colorbar(sc)
    plt.show()
