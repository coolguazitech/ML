import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
import numpy as np


def monitor_accords_number_of_samples(X, y, estimator, step_size=10, cv=10, n_jobs=1):
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator,
                                                            X=X,
                                                            y=y,
                                                            train_sizes=np.linspace(0.1, 1.0, step_size),
                                                            cv=cv,
                                                            n_jobs=n_jobs)
    train_mean = np.mean(train_scores, axis=-1)
    train_std = np.std(train_scores, axis=-1)
    test_mean = np.mean(test_scores, axis=-1)
    test_std = np.std(test_scores, axis=-1)

    plt.plot(train_sizes, train_mean, color='b', marker='o', markersize=5, label='training acc')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='b')
    plt.plot(train_sizes, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='validation acc')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.show()


def monitor_accords_parameters(X, y, estimator, param_name, param_range, cv=10):
    param_range = param_range
    train_scores, test_scores = validation_curve(estimator=estimator,
                                                 X=X,
                                                 y=y,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=cv,
                                                 )
    train_mean = np.mean(train_scores, axis=-1)
    train_std = np.std(train_scores, axis=-1)
    test_mean = np.mean(test_scores, axis=-1)
    test_std = np.std(test_scores, axis=-1)
    plt.plot(param_range, train_mean, color='b', marker='o', markersize=5, label='training acc')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='b')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='validation acc')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid()
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.ylim([0.8, 1.03])
    plt.show()
