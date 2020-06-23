from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from six import StringIO
from IPython.display import Image
import matplotlib.pyplot as plt
import pydotplus
import sys
import numpy as np

# Reference: https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py


def get_best_tree(data, test_split_percentage, criterion, name="", graph=False):
    source_vars = data[:, :-1]
    target_vars = data[:, len(data[0]) - 1]

    X_train, X_test, y_train, y_test = train_test_split(
        source_vars, target_vars, test_size=test_split_percentage/100, random_state=0)

    clf = DecisionTreeClassifier(random_state=0, criterion=criterion)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(
            random_state=0, criterion=criterion, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    # Remove tree with only one node
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]

    if graph:
        # Graph num nodes and depth vs alpha
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
        ax[0].set_xlabel("alpha")
        ax[0].set_ylabel("number of nodes")
        ax[0].set_title("Number of nodes vs alpha")
        ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
        ax[1].set_xlabel("alpha")
        ax[1].set_ylabel("depth of tree")
        ax[1].set_title("Depth vs alpha")
        fig.tight_layout()

        plot_name = "images/plot0_" + criterion + \
            ".png" if name == "" else "images/" + name + "_plot0_" + \
            criterion + "_" + str(test_split_percentage) + ".png"
        plt.savefig(plot_name)

        # Graph accuracy vs alpha
        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(ccp_alphas, train_scores, marker='o',
                label="train", drawstyle="steps-post")
        ax.plot(ccp_alphas, test_scores, marker='o',
                label="test", drawstyle="steps-post")
        ax.legend()

        plot_name = "images/plot1_" + criterion + \
            ".png" if name == "" else "images/" + name + "_plot1_" + \
            criterion + "_" + str(test_split_percentage) + ".png"
        plt.savefig(plot_name)

    i = test_scores.index(max(test_scores))
    return clfs[i], train_scores[i], test_scores[i], node_counts[i]


def visualize_tree(tree, test_split_percentage, feature_names, class_names, criterion, name=""):
    dot_data = StringIO()
    export_graphviz(
        tree,
        out_file=dot_data,
        filled=True,
        rounded=True,
        special_characters=True,
        feature_names=feature_names,
        class_names=class_names
    )

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    image = graph.create_png()

    plot_name = "images/plot2_" + criterion + \
        ".png" if name == "" else "images/" + name + "_plot2_" + \
        criterion + "_" + str(test_split_percentage) + ".png"

    with open(plot_name, 'wb') as f:
        f.write(image)


def main():
    if len(sys.argv) < 3:
        print('Usage: python main.py <test_split_percentage> <file_name>')
        exit()

    test_split_percentage = int(sys.argv[1])
    file_name = sys.argv[2]
    graph = '-g' in sys.argv

    out_name = ''
    if '-n' in sys.argv:
        out_name = sys.argv[sys.argv.index('-n') + 1]

    data = np.genfromtxt(file_name, delimiter=',')

    entropy_tree, entropy_train_scores, entropy_test_scores, entropy_node_counts = get_best_tree(
        data, test_split_percentage, criterion="entropy", graph=graph, name=out_name)
    gini_tree, gini_train_scores, gini_test_scores, gini_node_count = get_best_tree(
        data, test_split_percentage, criterion="gini", graph=graph, name=out_name)

    print('{} {} {}'.format(entropy_train_scores,
                            entropy_test_scores, entropy_node_counts))
    print('{} {} {}'.format(gini_train_scores, gini_test_scores, gini_node_count))

    if graph:
        if ('cleveland' in file_name):
            feature_names = ['age', 'sex', 'chest pain type', 'blood pressure', 'cholestoral', 'fasting blood sugar',
                             'ecg', 'max heart rate', 'exercise induced angina', 'oldpeak', 'slope', 'number of major vessels', 'thal']
            class_names = ['no heart disease', 'heart disease']
        else:
            feature_names = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over',
                             'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive',
                             'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
                             'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your',
                             'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl',
                             'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
                             'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
                             'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs',
                             'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu',
                             'word_freq_table', 'word_freq_conference', 'char_freq_:', 'char_freq_(', 'char_freq_[', 'char_freq_!',
                             'char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest',
                             'capital_run_length_total']
            class_names = ['not spam', 'spam']

        visualize_tree(entropy_tree, test_split_percentage, feature_names,
                       class_names, "entropy", name=out_name)
        visualize_tree(gini_tree, test_split_percentage, feature_names,
                       class_names, "gini", name=out_name)


if __name__ == "__main__":
    main()
