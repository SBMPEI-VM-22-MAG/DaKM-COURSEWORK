import numpy as np

class CustomDecisionTree:
    def __init__(self, max_depth=100, minimum_samples_split=2):
        self.maximum_depth = max_depth
        self.minimum_samples_split = minimum_samples_split
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
    
    def _build_dt(self, features, labels, current_depth=0):
        self.n_samples, self.n_features = features.shape
        self.n_class_labels = len(np.unique(labels))

        # Stopping criteria
        limit1 = current_depth >= self.maximum_depth
        limit2 = self.n_class_labels == 1
        limit3 = self.n_samples < self.minimum_samples_split
        if (limit1 or limit2 or limit3):
            unique_values, counts = np.unique(labels, return_counts=True)
            unique_counts = dict(zip(unique_values, counts))
            most_common_category = max(unique_counts, key=unique_counts.get)
            return CustomNode(value=most_common_category)
        
        # Best split
        random_features = np.random.choice(self.n_features, self.n_features, replace=False)
        
        best_feature = None
        best_threshold = None
        split_score = -1
        
        for feature in random_features:
            feature_values = features[:, feature]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                score = 0
                parent_entropy = self._entropy(labels)
                left_indices = np.argwhere(feature_values <= threshold).flatten()
                right_indices = np.argwhere(feature_values > threshold).flatten()
                n, n_left, n_right = len(labels), len(left_indices), len(right_indices)

                if (n_left != 0 and n_right != 0):
                    child_entropy = (n_left / n) * self._entropy(labels[left_indices]) + (n_right / n) * self._entropy(labels[right_indices])
                    score = parent_entropy - child_entropy
            
                if (score > split_score):
                    split_score = score
                    best_feature = feature
                    best_threshold = threshold
        
        # Recursive obtaining of children (nodes)
        left_indices = np.argwhere(features[:, best_feature] <= best_threshold).flatten()
        right_indices = np.argwhere(features[:, best_feature] > best_threshold).flatten()
        left_child = self._build_dt(features[left_indices, :], labels[left_indices], current_depth + 1)
        right_child = self._build_dt(features[right_indices, :], labels[right_indices], current_depth + 1)
        return CustomNode(best_feature, best_threshold, left_child, right_child)



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

