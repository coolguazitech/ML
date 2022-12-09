from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split

# The following objects are equipped with functions including fit and transform
# better to normalize data before fitting
# Sequential Backward Selection, SBS, supervised
class SBS:
    def __init__(self, estimator, n_components=1, test_size=0.25, random_state=1):
        self.estimator = estimator
        self.n_components = n_components
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        data = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        dim = data[0].shape[1]
        self.best_indices_ = tuple(range(dim))
        self.best_score = self._calc_score(*data, self.best_indices_)
        self.subsets = [self.best_indices_]
        self.scores = [self.best_score]

        while dim > self.n_components:
            subsets = []
            scores = []
            for p in combinations(self.best_indices_, dim - 1):
                subsets.append(p)
                scores.append(self._calc_score(*data, p))
            best_idx = np.argmax(scores)
            self.best_indices_, self.best_score = subsets[best_idx], scores[best_idx]
            self.subsets.append(self.best_indices_)
            self.scores.append(self.best_score)
            dim -= 1

        return self

    def transform(self, X):
        return X[:, self.best_indices_]
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def _calc_score(self, X_train, X_test, y_train, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        score = self.estimator.score(X_test[:, indices], y_test)
        return score

# other built-in extractors
# 1. Principal Component Analysis, PCA, unsupervised, linearly separable
# 3. Kernel Principal Component Analysis, KernelPCA, unsupervised, linearly inseparable
# 3. Linear Discriminant Analysis, LDA, supervised, linearly separable