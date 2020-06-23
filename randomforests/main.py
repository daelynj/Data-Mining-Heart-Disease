import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import sys
plot_num = 0


def plot_results(x, y, z, plot_name, version, test_split_percentage):
    global plot_num
    plot_num += 1

    plt.figure(plot_num)
    fig, ax = plt.subplots()
    ax.set_xlabel("number of trees")
    ax.set_ylabel("depth of tree")
    ax.set_title("Forest Size vs Tree Depth for " +
                 version.capitalize() + "ing Set")
    p = ax.contour(x, y, z, 20)
    plt.colorbar(p)

    plt.savefig('images/' + plot_name + '_' + version +
                '_' + str(test_split_percentage) + '.png')


def generate_forests(data, test_split_percentage, criterion, max_features, name):
    source_vars = data[:, :-1]
    target_vars = data[:, len(data[0]) - 1]

    X_train, X_test, y_train, y_test = train_test_split(
        source_vars, target_vars, test_size=test_split_percentage/100, random_state=0)

    estimators = [2, 10, 18, 26, 34, 42, 50, 58]
    depths = [1, 2, 3, 4, 5, 6]

    best_tree, best_score, best_train, best_attrs = None, 0, 0, []
    X, Y, Z, T = [], [], [], []
    for estimator in estimators:
        x, y, z, t = [], [], [], []
        for depth in depths:
            clf = RandomForestClassifier(
                criterion=criterion,
                n_estimators=estimator,
                max_depth=depth,
                max_features="sqrt",
                random_state=0
            )
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scr = clf.score(X_train, y_train)

            if (score > best_score):
                best_score = score
                best_tree = clf
                best_attrs = [estimator, depth]
                best_train = scr

            x.append(estimator)
            y.append(depth)
            z.append(score)
            t.append(clf.score(X_train, y_train))

        X.append(x)
        Y.append(y)
        Z.append(z)
        T.append(t)

    plot_name = name + '_' + criterion
    plot_results(X, Y, Z, plot_name, 'test', test_split_percentage)
    plot_results(X, Y, T, plot_name, 'train', test_split_percentage)

    return best_tree, best_score, best_attrs, best_train


def get_best_forest(data, test_split_percentage, criterion, name):

    best_forest_sqrt, best_score_sqrt, best_attrs_sqrt, best_train = generate_forests(
        data, test_split_percentage, criterion, 'sqrt', name)
    # best_forest_none, best_score_none, best_attrs_none = generate_forests(data, criterion, None, name)

    # if best_score_sqrt > best_score_none:
    return best_forest_sqrt, best_score_sqrt, best_attrs_sqrt, best_train
    # return best_forest_none, best_score_none, best_attrs_none.append('none')


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py <test_split_percentage> <file_name> -n <out_name>')
        exit()

    test_split_percentage = int(sys.argv[1])
    file_name = sys.argv[2]

    out_name = ''
    if '-n' in sys.argv:
        out_name = sys.argv[sys.argv.index('-n') + 1]

    data = np.genfromtxt(file_name, delimiter=',')

    best_tree_gini, best_score_gini, best_attrs_gini, best_train_gini = get_best_forest(
        data, test_split_percentage, criterion="gini", name=out_name)
    best_tree_entr, best_score_entr, best_attrs_entr, best_train_entr = get_best_forest(
        data, test_split_percentage, criterion="entropy", name=out_name)

    print('{} {} {}'.format(best_train_entr, best_score_entr, best_attrs_entr))
    print('{} {} {}'.format(best_train_gini, best_score_gini, best_attrs_gini))


if __name__ == "__main__":
    main()
