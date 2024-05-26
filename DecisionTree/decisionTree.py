import numpy as np

class DecisionNode:
    def __init__(self,feature,value):
        self.featureName = feature
        self.selfValue = value
        self.leftTree = None
        self.rightTree = None


class LeafNode:
    def __init__(self, value):
        self.value = value


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(set(y)) == 1:
            return self._leaf_node(y)

        best_feature, best_value = self._find_best_split(X, y)
        if best_feature is None or best_value is None:
            return self._leaf_node(y)

        mask = X[:, best_feature] <= best_value
        left_X, left_y = X[mask], y[mask]
        right_X, right_y = X[~mask], y[~mask]

        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)

        decision_node = DecisionNode(best_feature, best_value)
        decision_node.left = left_subtree
        decision_node.right = right_subtree

        return decision_node

    def _find_best_split(self, X, y):
        best_feature, best_value, best_score = None, None, -np.inf

        for feature in range(self.n_features):
            values = np.unique(X[:, feature])
            for value in values:
                score = self._gini_index(X, y, feature, value)
                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_value = value

        return best_feature, best_value

    def _gini_index(self, X, y, feature, value):
        mask = X[:, feature] <= value
        n_left = np.sum(mask)
        n_right = np.sum(~mask)

        gini_left = self._gini_impurity(y[mask])
        gini_right = self._gini_impurity(y[~mask])

        gini_index = (n_left * gini_left + n_right * gini_right) / len(y)
        return gini_index

    def _gini_impurity(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        gini = 1 - np.sum(probs ** 2)
        return gini

    def _leaf_node(self, y):
        counts = np.bincount(y)
        label = np.argmax(counts)
        return LeafNode(label)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if isinstance(node, LeafNode):
            return node.label

        if x[node.feature] <= node.value:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
        
