import numpy as np

# Класс узла ДР
class CustomNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


# # Класс ДР    
class CustomDecisionTree():
    # критерии остановки: max_depth, min_samples_split, root_node
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _is_finished(self, depth):
        # вычисляем проверки
        limit1 = depth >= self.max_depth
        limit2 = self.n_class_labels == 1
        limit3 = self.n_samples < self.min_samples_split
        if(limit1 or limit2 or limit3):
            return True
        return False

    def _unique_counts(self, y):
        unique_values, counts = np.unique(y, return_counts=True)
        # используем функцию zip, чтобы объединить массив unique_values и массив counts в кортежи,
        # где каждый кортеж содержит уникальное значение и количество его повторений
        # dict() используется для создания словаря из кортежей
        return dict(zip(unique_values, counts))

    def _entropy(self, y):
        n = len(y)
        unique_counts = self._unique_counts(y)
        entropy = 0.0
        for count in unique_counts.values():
            p = count / n
            entropy -= p * np.log2(p)
        return entropy

    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if (n_left == 0 or n_right == 0):
            return 0
        
        child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        split = {"score": -1, "feat": None, "thresh": None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if (score > split["score"]):
                    split["score"] = score
                    split["feat"] = feat
                    split["thresh"] = thresh

        return split["feat"], split["thresh"]

    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # критерий остановки
        if (self._is_finished(depth)):
            most_common_Label = max(self._unique_counts(y), key=self._unique_counts(y).get)
            return CustomNode(value=most_common_Label)
        
        # лучшее разделение
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # рекурсивное получение потомков (узлов)
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return CustomNode(best_feat, best_thresh, left_child, right_child)

    def _traverse_tree(self, x, node):
        if (node.is_leaf()):
            return node.value
        if (x[node.feature] <= node.threshold):
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()
        self.root = self._build_tree(X, y)

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    # структура дерева
    def print_tree_structure(self, node=None, depth=0):
        if node is None:
            node = self.root

        indent = "  " * depth
        if node.is_leaf():
            print(indent + f"Leaf Node: Class {node.value}")
        else:
            print(indent + f"Decision Node: Feature {node.feature}, Threshold {node.threshold}")
            print(indent + "  Left Branch:")
            self.print_tree_structure(node.left, depth + 1)
            print(indent + "  Right Branch:")
            self.print_tree_structure(node.right, depth + 1)
