import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


class DecisionTree():
    def __init__(self, min_split=2, max_depth=100):
        self.root = None
        self.min_split = min_split
        self.max_depth = max_depth

    def tree(self, data, cur_depth=0):
        X, Y = data[:, :-1], data[:, -1]
        num_samples, num_features = np.shape(X)
        if num_samples >= self.min_split and cur_depth <= self.max_depth:
            best_split = self.best_split(data, num_samples, num_features)
            if best_split["info_gain"] > 0:
                left_split = self.tree(best_split["dataset_left"], cur_depth + 1)
                right_split = self.tree(best_split["dataset_right"], cur_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_split, right_split, best_split["info_gain"])
        leaf_value = self.leaf_value(Y)
        return Node(value=leaf_value)

    def best_split(self, data, num_samples, num_features):
        best_split_dic = {}
        max_info_gain = -float("inf")
        for feature_index in range(num_features):
            feature_values = data[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(data, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = data[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    if curr_info_gain > max_info_gain:
                        best_split_dic["feature_index"] = feature_index
                        best_split_dic["threshold"] = threshold
                        best_split_dic["dataset_left"] = dataset_left
                        best_split_dic["dataset_right"] = dataset_right
                        best_split_dic["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split_dic

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + data.columns[tree.feature_index], "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.tree(dataset)

    def predict(self, X):
        predictions = [self.prediction(x, self.root) for x in X]
        return predictions

    def prediction(self, x, tree):
        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.prediction(x, tree.left)
        else:
            return self.prediction(x, tree.right)

data = pd.read_csv('contact-lenses.csv')

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

classifier = DecisionTree()
classifier.fit(X_train,Y_train)
classifier.print_tree()

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test,Y_pred))