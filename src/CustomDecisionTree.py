import numpy as np

class CustomDecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _entropy(self, y):
        n = len(y)
        unique_values, counts = np.unique(y, return_counts=True)
        entropy = 0.0
        for count in counts:
            p = count / n
            entropy -= p * np.log2(p)
        return entropy

    def _find_best_split(self, X, y, available_features):
        best_split = {"score": -1, "feature": None, "threshold": None}

        for feature in available_features:
            feature_values = X[:, feature]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_indices = np.argwhere(feature_values <= threshold).flatten()
                right_indices = np.argwhere(feature_values > threshold).flatten()
                n, n_left, n_right = len(y), len(left_indices), len(right_indices)

                if n_left == 0 or n_right == 0:
                    continue

                left_entropy = self._entropy(y[left_indices])
                right_entropy = self._entropy(y[right_indices])

                child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
                information_gain = self._entropy(y) - child_entropy

                if information_gain > best_split["score"]:
                    best_split["score"] = information_gain
                    best_split["feature"] = feature
                    best_split["threshold"] = threshold

        return best_split

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        if depth >= self.max_depth or num_classes == 1 or num_samples < self.min_samples_split:
            most_common_label = np.bincount(y).argmax()
            return {"value": most_common_label, "is_leaf": True}

        available_features = np.random.choice(num_features, num_features, replace=False)
        best_split = self._find_best_split(X, y, available_features)

        left_indices = np.argwhere(X[:, best_split["feature"]] <= best_split["threshold"]).flatten()
        right_indices = np.argwhere(X[:, best_split["feature"]] > best_split["threshold"]).flatten()

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_subtree,
            "right": right_subtree,
            "is_leaf": False,
        }

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()
        self.root = self._build_tree(X, y)

    def _predict_one(self, x, node):
        if node["is_leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        return [self._predict_one(x, self.root) for x in X]