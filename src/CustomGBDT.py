import numpy as np
from CustomDecisionTree import CustomDecisionTree

class CustomGBDT:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []  # храним базовые модели (деревья)
        
    def fit(self, X, y):
        # инициализируем композицию предсказаний нулевым вектором
        predictions = np.zeros(len(y))
        # print("predictions", predictions)
        idx = 0
        for _ in range(self.n_estimators):
            # вычисляем градиент
            gradient = y - predictions
            # print("gradient")
            # print(gradient)
            idx += 1
            print("Tree=",idx)
            
            # обучаем дерево на градиенте
            tree = CustomDecisionTree(max_depth=self.max_depth)
            tree.fit(X, gradient)  # градиент вместо меток
            
            # вычисляем прогнозы базовой модели
            tree_predictions = tree.predict(X)
            
            # обновляем композицию с учетом learning_rate
            predictions += self.learning_rate * tree_predictions
            
            # добавляем дерево в список
            self.models.append(tree)
    
    def predict(self, X):
        # для прогноза суммируем прогнозы всех деревьев
        predictions = np.zeros(len(X))
        for model in self.models:
            tree_predictions = model.predict(X)
            predictions += self.learning_rate * tree_predictions
        
        # преобразуем предсказания в бинарные метки классов (0 и 1)
        return np.where(predictions >= 0.5, 1, 0)
