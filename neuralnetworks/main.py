from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import numpy as np

plot_num = 0


def generate_networks(data, solver, name, test_split_percentage):
    print(solver)

    source_vars = data[:, :-1]
    target_vars = data[:, len(data[0]) - 1]

    X_train, X_test, y_train, y_test = train_test_split(
        source_vars, target_vars, test_size=test_split_percentage/100, random_state=0)

    sizes = list(range(3, 40, 2))

    clfs = []
    for size in sizes:
        clf = MLPClassifier(
            solver=solver,
            hidden_layer_sizes=size,
            alpha=0.00001,
            learning_rate='constant',
            random_state=0,
            max_iter=400
        )
        clf.fit(X_train, y_train)
        clfs.append(clf)

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("number of nodes in hidden layer")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs Size of Hidden Layer for training and test sets")
    ax.plot(sizes, train_scores, marker='o', label="train")
    ax.plot(sizes, test_scores, marker='o', label="test")
    ax.legend()

    plot_name = name + '_' + solver
    plt.savefig('images/' + plot_name + '.png')

    i = test_scores.index(max(test_scores))

    return clfs[i], test_scores[i], sizes[i], train_scores[i]


def get_best_network(data, name, test_split_percentage):
    best_network_sgd, best_score_sgd, best_attrs_sgd, tr_sgd = generate_networks(
        data, 'sgd', name, test_split_percentage)
    best_network_adam, best_score_adam, best_attrs_adam, tr_adam = generate_networks(
        data, 'adam', name, test_split_percentage)

    print('{} {}'.format(tr_sgd, best_score_sgd))
    print('{} {}'.format(tr_adam, best_score_adam))

    if best_score_sgd > best_score_adam:
        return best_network_sgd, best_score_sgd, best_attrs_sgd, 'sgd'
    return best_network_adam, best_score_adam, best_attrs_adam, 'adam'


def main():
    if len(sys.argv) < 3:
        print('Usage: python main.py <test_split_percentage> <file_name> -n <out_name>')
        exit()

    test_split_percentage = int(sys.argv[1])
    file_name = sys.argv[2]

    out_name = ''
    if '-n' in sys.argv:
        out_name = sys.argv[sys.argv.index('-n') + 1]

    data = np.genfromtxt(file_name, delimiter=',')

    network, score, size, solver = get_best_network(
        data, out_name, test_split_percentage)


if __name__ == "__main__":
    main()
