import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
import numpy as np

def monitor_size_of_dataset(X, y, estimator, step=10, cv=10, n_jobs=1):
    """Determine the size of dataset by monitoring the learning curve."""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, step),
        cv=cv,
        n_jobs=n_jobs
    )
    train_mean = train_scores.mean(axis=-1)
    train_std = train_scores.std(axis=-1)
    test_mean = test_scores.mean(axis=-1)
    test_std = test_scores.std(axis=-1)
    
    plt.plot(train_sizes, train_mean, color='darkorange', marker='o', markersize=5, label='training acc')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='darkorange')
    plt.plot(train_sizes, test_mean, color='royalblue', linestyle='-.', marker='o', markersize=5, label='validation acc')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='royalblue')
    plt.grid()
    plt.title(f'Learning Curve of {estimator.__class__.__name__}')
    plt.xlabel('size of dataset')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.ylim([0.8, 1.03])
    plt.show()

def monitor_parameter(X, y, estimator, param_name, param_range, cv=10):
    """Determine the parameter of estimator by monitoring the learning curve."""
    train_scores, test_scores = validation_curve(
        estimator=estimator,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
    )
    train_mean = train_scores.mean(axis=-1)
    train_std = train_scores.std(axis=-1)
    test_mean = test_scores.mean(axis=-1)
    test_std = test_scores.std(axis=-1)
    plt.semilogx(param_range, train_mean, color='darkorange', marker='o', markersize=5, label='training acc')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='darkorange')
    plt.semilogx(param_range, test_mean, color='royalblue', linestyle='-.', marker='o', markersize=5, label='validation acc')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='royalblue')
    plt.grid()
    plt.title(f'Learning Curve of {estimator.__class__.__name__}')
    plt.xlabel(f'parameter {param_name}')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.ylim([0.8, 1.03])
    plt.show()