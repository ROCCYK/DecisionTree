import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class Node(): # Created a Node class to store tree value.
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # Initialized the self variables and set it to none for now.
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


class DecisionTree(): # Created a Decision Tree Class.
    def __init__(self, min_split=2, max_depth=100):
        # Initialized the self variables, set the root to none and set the default values of min split to 2 and
        # max depth to 100.
        self.root = None
        self.min_split = min_split
        self.max_depth = max_depth

    def tree(self, data, cur_depth=0): # Created the tree function that takes in the numpy array and set current depth to 0.
        X, Y = data[:, :-1], data[:, -1] # Excluded the last column for X and set Y as the last column.
        num_samples, num_features = np.shape(X) # num of samples as the num of rows and num of features as num of columns of the variable X.
        if num_samples >= self.min_split and cur_depth <= self.max_depth: # check to make sure there's enough samples
            # to split and make sure we don't exceed the max depth we set.
            best_split = self.best_split(data, num_samples, num_features) # assigned the best split dictionary to best_split.
            if best_split["info_gain"] > 0: # check to make sure the best split's information gain is greater than 0.
                # Recursively call the tree function on the best split's left and right dataset and add 1 to current depth.
                left_split = self.tree(best_split["dataset_left"], cur_depth + 1)
                right_split = self.tree(best_split["dataset_right"], cur_depth + 1)
                # store the best split's feature index (column index), threshold (column value), left split, right split,
                # and information gain in the Node class.
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_split, right_split, best_split["info_gain"])
        leaf_value = self.leaf_value(Y) # find the leaf value of Y column and assigned it to leaf_value.
        return Node(value=leaf_value) # store the leaf value in the Node class.

    def best_split(self, data, num_samples, num_features):# Created the best_split function that takes in the dataset, num of samples and num of features.
        best_split_dic = {} # Initialized the best split dictionary.
        max_info_gain = -float("inf") # Initialized max information gain as negative infinity so that we can replace it with the dataset's IG.
        for feature_index in range(num_features): # run a for loop on the X columns.
            feature_values = data[:, feature_index] # gets all the values of the column.
            possible_thresholds = np.unique(feature_values) # gets all the possible values of that column.
            for threshold in possible_thresholds: # run a for loop on all the possible values.
                dataset_left, dataset_right = self.split(data, feature_index, threshold) # runs the split function and stores the left and right split of the dataset.
                if len(dataset_left) > 0 and len(dataset_right) > 0: # makes sure that there are values in the left and right dataset.
                    y, left_y, right_y = data[:, -1], dataset_left[:, -1], dataset_right[:, -1] # gets the y column of the parent and child splits.
                    curr_info_gain = self.information_gain(y, left_y, right_y) # calculates the current information gain.
                    # calculates the IG for all thresholds and keeps replacing the max information gain to find out which is the best threshold to split on.
                    if curr_info_gain > max_info_gain:
                        best_split_dic["feature_index"] = feature_index
                        best_split_dic["threshold"] = threshold
                        best_split_dic["dataset_left"] = dataset_left
                        best_split_dic["dataset_right"] = dataset_right
                        best_split_dic["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split_dic # returns the best split threshold dictionary.

    def split(self, dataset, feature_index, threshold): # created the split function that takes in the dataset, feature index and threshold.
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold]) # left split is when feature is less than or equal to the threshold.
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold]) # right split is when feature greater than the threshold.
        return dataset_left, dataset_right # returns the left and right split.

    def information_gain(self, parent, l_child, r_child): # created the information gain function that takes in the parent, left, and right child.
        # used the information gain formula with entropy to calculate.
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain # return information gain.

    def entropy(self, y): # created the entropy function that takes in the y column.
        class_labels = np.unique(y) # list of possible values of the y column.
        entropy = 0 # initialized the entropy to 0 for us to be able to +=.
        # used the entropy formula to calculate.
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy # return entropy.

    def leaf_value(self, Y): # Created the leaf_value function that takes in the Y column.
        Y = list(Y) # set Y as a list.
        return max(Y, key=Y.count) # returned the max counted Y.

    def print_tree(self, tree=None, indent=" "): # created the print_tree function to visualize the tree.
        if not tree: # if tree = none.
            tree = self.root # we set tree as self.root.
        if tree.value is not None: # if the value is not empty we just print the tree value.
            print(tree.value)
        else:
            # prints the tree from the tree function and the Node class' self variables.
            print("X_" + data.columns[tree.feature_index], "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y): # created the fit function to pass in our X train, and Y train.
        dataset = np.concatenate((X, Y), axis=1) # combined the X train and Y train to a full dataset with X and Y.
        self.root = self.tree(dataset) # set the Node's self.root to the tree function with the combined dataset.

    def predict(self, X): # created the predict function that takes in the X test.
        predictions = [self.prediction(x, self.root) for x in X] # calls the prediction function to predict the Y and assigns the value to predictions.
        return predictions # returns the predictions list.

    def prediction(self, x, tree): # created the prediction function that takes in the x features array and the self.root tree.
        if tree.value != None: return tree.value # if the tree's values is not empty we just return the tree value.
        feature_val = x[tree.feature_index] # assigns feature_val to the x array's feature index from the tree it built.
        # recursively calls the prediction function until we get a leaf node and
        # assigns left and right split depending if the feature value is less than or equal to the tree's threshold value.
        if feature_val <= tree.threshold:
            return self.prediction(x, tree.left)
        else:
            return self.prediction(x, tree.right)

# Tests
data = pd.read_csv('lenses.csv')

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)

classifier = DecisionTree()
classifier.fit(X_train,Y_train)
classifier.print_tree()

Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test,Y_pred))