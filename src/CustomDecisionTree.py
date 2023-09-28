import numpy as np

class CustomDecisionTree:
    def __init__(self, max_depth=100, minimum_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = minimum_samples_split
        self.root = None
        
    def fit(self, features, labels):
        self.root = self._build_dt(features, labels)

    def predict(self, features):
        predictions = [self._run_tree(x, self.root) for x in features]
        return np.array(predictions)

    def display_tree(self, node=None, current_depth=0):
        if node is None:
            node = self.root

        indentation = "  " * current_depth
        if node.is_leaf():
            print(indentation + f"Secret Leaf Node: Category {node.value}")
        else:
            print(indentation + f"Enigmatic Node: Feature {node.feature}, Threshold {node.threshold}")
            print(indentation + "  Left Branch:")
            self.display_tree(node.left, current_depth + 1)
            print(indentation + "  Right Branch:")
            self.display_tree(node.right, current_depth + 1)

    def _entropy(self, labels):
        n = len(labels)
        unique_values, counts = np.unique(labels, return_counts=True)
        unique_counts = dict(zip(unique_values, counts))
        entropy = 0.0
        for count in unique_counts.values():
            p = count / n
            entropy -= p * np.log2(p)
        return entropy
    
    def _build_dt(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        limit1 = depth >= self.max_depth
        limit2 = self.n_class_labels == 1
        limit3 = self.n_samples < self.min_samples_split
        if (limit1 or limit2 or limit3):
            unique_values, counts = np.unique(y, return_counts=True)
            u_c = dict(zip(unique_values, counts))
            most_common_Label = max(u_c, key=u_c.get)
            return CustomNode(value=most_common_Label)
        
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat = None
        best_thresh = None
        split = -1
        for feat in rnd_feats:
            # print(feat)
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = 0
                parent_loss = self._entropy(y)
                left_idx = np.argwhere(X_feat <= thresh).flatten()
                right_idx = np.argwhere(X_feat > thresh).flatten()
                n, n_left, n_right = len(y), len(left_idx), len(right_idx)

                if (n_left != 0 and n_right != 0):
                    child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
                    score = parent_loss - child_loss
            
                if (score > split):
                    split = score
                    best_feat = feat
                    best_thresh = thresh
        
        # рекурсивное получение потомков (узлов)
        left_idx = np.argwhere(X[:, best_feat] <= best_thresh).flatten()
        right_idx = np.argwhere(X[:, best_feat] > best_thresh).flatten()
        left_child = self._build_dt(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_dt(X[right_idx, :], y[right_idx], depth + 1)
        return CustomNode(best_feat, best_thresh, left_child, right_child)



    def _run_tree(self, x, node):
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


# Node Class
class CustomNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature, self.threshold, self.left, self.right, self.value = feature, threshold, left, right, value

    def is_leaf(self):
        return self.value is not None