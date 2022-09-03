from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from colorama import Fore
from sklearn import tree

import visualize
import MLP


def decision_tree(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    acc_t = accuracy_score(y_test, clf.predict(X_test))
    report = classification_report(y_test, clf.predict(X_test), labels=[0, 1])
    confusion = confusion_matrix(y_test, clf.predict(X_test))
    accuracy('Decision tree', acc_t, report, confusion)
    # visualize_data(X_test, y_test, clf.predict(X_test))


def naiive_base(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    acc_t = accuracy_score(y_test, gnb.predict(X_test))
    report = classification_report(y_test, gnb.predict(X_test), labels=[0, 1])
    confusion = confusion_matrix(y_test, gnb.predict(X_test))
    accuracy('Naiive base', acc_t, report, confusion)


def logistic_regression(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    acc_t = accuracy_score(y_test, clf.predict(X_test))
    report = classification_report(y_test, clf.predict(X_test), labels=[0, 1])
    confusion = confusion_matrix(y_test, clf.predict(X_test))
    accuracy('Logistic Regression', acc_t, report, confusion)


def knn(X_train, X_test, y_train, y_test):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    acc_t = accuracy_score(y_test, neigh.predict(X_test))
    report = classification_report(y_test, neigh.predict(X_test), labels=[0, 1])
    confusion = confusion_matrix(y_test, neigh.predict(X_test))
    accuracy('KNN', acc_t, report, confusion)


def svm(X_train, X_test, y_train, y_test):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    acc_t = accuracy_score(y_test, clf.predict(X_test))
    report = classification_report(y_test, clf.predict(X_test), labels=[0, 1])
    confusion = confusion_matrix(y_test, clf.predict(X_test))
    accuracy('SVM', acc_t, report, confusion)


def mlp_sklearn(X_train, X_test, y_train, y_test):
    clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=(15, 12)
                        , alpha=1e-5).fit(X_train, y_train)
    acc_t = accuracy_score(y_test, clf.predict(X_test))
    report = classification_report(y_test, clf.predict(X_test), labels=[0, 1])
    confusion = confusion_matrix(y_test, clf.predict(X_test))
    accuracy('MLP sklearn', acc_t, report, confusion)


def mlp_pytorch(info_list, accept_train):
    train_dl, test_dl = MLP.prepare_data(info_list, accept_train)
    X_test, y_test, y_pre = [], [], []
    for i, (inputs, targets) in enumerate(test_dl):
        for t in targets:
            y_test.append(t.tolist()[0])
        for j in inputs:
            X_test.append(j.tolist())
    model = MLP.MLP(len(info_list[0]))
    MLP.train_model(train_dl, model)
    y_prediction = MLP.evaluate_model(train_dl, test_dl, model)
    for y in y_prediction:
        y_pre.append(y[0])
    acc_t = accuracy_score(y_test, y_pre)
    report = classification_report(y_test, y_pre, labels=[0, 1])
    confusion = confusion_matrix(y_test, y_pre)
    accuracy('MLP pytorch', '%.3f' % acc_t, report, confusion)
    visualize_data(X_test, y_test, y_pre)


def accuracy(title, acc_t, report, confusion):
    print(Fore.RED + title)
    print(Fore.WHITE + 'report:\n', report)
    print('confusion matrix:\n',confusion, '\n')


def visualize_data(X, y_true, y_pre):
    # visualize.visualize_acc(info, accept, info_list)
    # visualize.visualize_count(info)
    visualize.visualize_3d(X, y_true, y_pre)
    # visualize.visualize_truth(X, y_true, y_pre)