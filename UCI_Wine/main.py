import sklearn
from sklearn import datasets
from my_utils.feature_selection import PCA
from my_utils import pickle_model
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from my_utils import feature_selection
from my_utils import visualization

df = pd.read_csv(r'C:\Users\ASUS ROG\PycharmProjects\Machine Learning\UCI_Wine\wine.data', header=None)
columns = ['classlabel', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
           'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
           'OD280/OD315 of diluted wines', 'Proline']
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

########################################################################################################################
# choose model(s) (2x5 CV)                                                                                             #
########################################################################################################################
# SVC
pipe_svc = Pipeline([['sc', StandardScaler()], ['pca', PCA()], ['svc', SVC(random_state=1)]])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear'], 'pca__n_components': list(range(1, X_train.shape[1]))},
              {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf'],
               'pca__n_components': list(range(1, X_train.shape[1]))}]
gs = GridSearchCV(estimator=pipe_svc,
                    param_grid=param_grid,
                    cv=2,
                    scoring='accuracy',
                    n_jobs=-1)
gs.fit(X_train, y_train)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
print('CV 2x5 accuracy: %.2f +- %.2f' % (np.mean(scores), np.std(scores)))
print(gs.best_params_)
# # result:
# # CV 2x5 accuracy: 0.96 +- 0.01
# # {'pca__n_components': 12, 'svc__C': 1000.0, 'svc__gamma': 0.001, 'svc__kernel': 'rbf'}
# RandomForest
pipe_rf = Pipeline([['pca', PCA()], ['rf', RandomForestClassifier(random_state=1)]])
param_grid = [{'rf__n_estimators': list(range(10, 15)),
               'rf__criterion': ['gini', 'entropy'],
               'rf__max_depth': list(range(5, 10)),
               'pca__n_components': list(range(1, X_train.shape[1]))}]
gs_2 = GridSearchCV(estimator=pipe_rf,
                    param_grid=param_grid,
                    cv=2,
                    scoring='accuracy',
                    n_jobs=-1)
gs_2.fit(X_train, y_train)
scores = cross_val_score(gs_2, X_train, y_train, scoring='accuracy', cv=5)
print('CV 2x5 accuracy: %.2f +- %.2f' % (np.mean(scores), np.std(scores)))
print(gs_2.best_params_)
# result:
# CV 2x5 accuracy: 0.92 +- 0.02
# {'pca__n_components': 27, 'rf__criterion': 'gini', 'rf__max_depth': 7, 'rf__n_estimators': 11}
# LR
pipe_lr = Pipeline([['sc', StandardScaler()], ['pca', PCA()], ['lr', LogisticRegression(random_state=1)]])
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'lr__C': param_range, 'pca__n_components': list(range(1, X_train.shape[1]))}]
gs_3 = GridSearchCV(estimator=pipe_lr,
                    param_grid=param_grid,
                    cv=2,
                    scoring='accuracy',
                    n_jobs=-1)
gs_3.fit(X_train, y_train)
scores = cross_val_score(gs_3, X_train, y_train, scoring='accuracy', cv=5)
print('CV 2x5 accuracy: %.2f +- %.2f' % (np.mean(scores), np.std(scores)))
print(gs_3.best_params_)
# result:
# CV 2x5 accuracy: 0.97 +- 0.02
# {'lr__C': 0.1, 'lr__penalty': 'l2', 'pca__n_components': 8}
########################################################################################################################
# build model (if haven't chosen)                                                                                      #
########################################################################################################################
# pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
########################################################################################################################
# brute_force (adjust parameters)                                                                                      #
########################################################################################################################
# # SVC
# pipe_svc = Pipeline([['sc', StandardScaler()], ['pca', PCA()], ['svc', SVC(random_state=1)]])
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear'], 'pca__n_components': list(range(1, X_train.shape[1]))},
#               {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf'],
#                'pca__n_components': list(range(1, X_train.shape[1]))}]
# gs_1 = GridSearchCV(estimator=pipe_svc,
#                     param_grid=param_grid,
#                     cv=10,
#                     scoring='accuracy',
#                     n_jobs=-1)
# gs_1.fit(X_train, y_train)
# best_model_1 = gs_1.best_estimator_
# print(gs_1.best_params_)
# print('model_1 accuracy: %.2f' % gs_1.best_score_)
# # RF
# pipe_rf = Pipeline([['pca', PCA()], ['rf', RandomForestClassifier(random_state=1)]])
# param_grid = [{'rf__n_estimators': list(range(10, 15)),
#                'rf__criterion': ['gini', 'entropy'],
#                'rf__max_depth': list(range(5, 10)),
#                'pca__n_components': list(range(1, X_train.shape[1]))}]
# gs_2 = GridSearchCV(estimator=pipe_rf,
#                     param_grid=param_grid,
#                     cv=10,
#                     scoring='accuracy',
#                     n_jobs=-1)
# gs_2.fit(X_train, y_train)
# best_model_2 = gs_2.best_estimator_
# print(gs_2.best_params_)
# print('model_2 accuracy: %.2f' % gs_2.best_score_)
# # LR
# pipe_lr = Pipeline([['sc', StandardScaler()], ['pca', PCA()], ['lr', LogisticRegression(random_state=1)]])
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# param_grid = [{'lr__C': param_range, 'pca__n_components': list(range(1, X_train.shape[1]))}]
# gs_3 = GridSearchCV(estimator=pipe_lr,
#                     param_grid=param_grid,
#                     cv=10,
#                     scoring='accuracy',
#                     n_jobs=-1)
# gs_3.fit(X_train, y_train)
# best_model_3 = gs_3.best_estimator_
# print(gs_3.best_params_)
# print('model_3 accuracy: %.2f' % gs_3.best_score_)
########################################################################################################################
# ensemble                                                                                                             #
########################################################################################################################
# clf_1 = Pipeline([['sc', StandardScaler()], ['pca', PCA(3)], ['svc', SVC(random_state=1, C=1.0, gamma=0.01,
#                                                                          kernel='rbf')]])
# clf_2 = Pipeline([['pca', PCA(7)], ['rf', RandomForestClassifier(random_state=1, criterion='entropy', max_depth=6,
#                                                                  n_estimators=10)]])
# clf_3 = Pipeline([['sc', StandardScaler()], ['pca', PCA(7)], ['lr', LogisticRegression(random_state=1, C=0.01)]])
# clf_e = VotingClassifier(estimators=[('svc', clf_1), ('rf', clf_2), ('lr', clf_3)], voting='hard')
#
# best_score = 0.0
# best_model = None
# for clf, label in zip([clf_1, clf_2, clf_3, clf_e], ['SVC', 'RF', 'LR', 'Ensemble']):
#     scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=10)
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#     if scores.mean() >= best_score:
#         best_score = scores.mean()
#         best_model = clf
# pickle_model.save_model(r'C:\Users\ASUS ROG\PycharmProjects\Machine '
#                         r'Learning\UCI_Wine\models\UCI_Wine_1-1.pickle', best_model)
# result:
# Accuracy: 0.93 (+/- 0.07) [SVC]
# Accuracy: 0.94 (+/- 0.07) [RF]
# Accuracy: 0.98 (+/- 0.05) [LR]
# Accuracy: 0.96 (+/- 0.06) [Ensemble]
########################################################################################################################
# fit on whole train_data and scoring on test_data                                                                     #
########################################################################################################################
model = pickle_model.load_model(r'C:\Users\ASUS ROG\PycharmProjects\Machine '
                                r'Learning\UCI_Wine\models\UCI_Wine_1-1.pickle')
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print('Test accuracy: %.4f' % acc)

# Test accuracy: 0.9630


