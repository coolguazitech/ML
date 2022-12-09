from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing
import sklearn
from sklearn.preprocessing import StandardScaler
from my_utils.feature_selection import PCA
from my_utils import pickle_model
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

data = pd.read_csv("car.data")
# print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(data["buying"].values)  # fit -> transform
maint = le.fit_transform(data["maint"].values)
doors = le.fit_transform(data["doors"].values)
persons = le.fit_transform(data["persons"].values)
lug_boot = le.fit_transform(data["lug_boot"].values)
safety = le.fit_transform(data["safety"].values)
cls = le.fit_transform(data["class"].values)

predict = "class"

X = np.array(list(zip(buying, maint, doors, persons, lug_boot, safety)))
y = np.array(cls)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=1,
                                                                            stratify=y)

########################################################################################################################
# choose model(s) (2x5 CV)                                                                                             #
########################################################################################################################
# # SVC
# pipe_svc = Pipeline([['sc', StandardScaler()], ['pca', PCA()], ['svc', SVC(random_state=1)]])
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear'],
#                'pca__n_components': list(range(1, X_train.shape[1] + 1))},
#               {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf'],
#                'pca__n_components': list(range(1, X_train.shape[1] + 1))}]
# gs = GridSearchCV(estimator=pipe_svc,
#                   param_grid=param_grid,
#                   cv=2,
#                   scoring='accuracy',
#                   n_jobs=-1)
# gs.fit(X_train, y_train)
# scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
# print('CV 2x5 accuracy: %.2f +- %.2f' % (np.mean(scores), np.std(scores)))
# print(gs.best_params_)
# # RF
# pipe_rf = Pipeline([['pca', PCA()], ['rf', RandomForestClassifier(random_state=1)]])
# param_grid = [{'rf__n_estimators': list(range(5, 15)),
#                'rf__criterion': ['gini', 'entropy'],
#                'rf__max_depth': list(range(5, 10)),
#                'pca__n_components': list(range(1, X_train.shape[1] + 1))}]
# gs = GridSearchCV(estimator=pipe_rf,
#                   param_grid=param_grid,
#                   cv=2,
#                   scoring='accuracy',
#                   n_jobs=-1)
# gs.fit(X_train, y_train)
# scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
# print('CV 2x5 accuracy: %.2f +- %.2f' % (np.mean(scores), np.std(scores)))
# print(gs.best_params_)
# # LR
# pipe_lr = Pipeline([['sc', StandardScaler()], ['pca', PCA()], ['lr', LogisticRegression(random_state=1)]])
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# param_grid = [{'lr__C': param_range, 'pca__n_components': list(range(1, X_train.shape[1] + 1))}]
# gs = GridSearchCV(estimator=pipe_lr,
#                   param_grid=param_grid,
#                   cv=2,
#                   scoring='accuracy',
#                   n_jobs=-1)
# gs.fit(X_train, y_train)
# scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
# print('CV 2x5 accuracy: %.2f +- %.2f' % (np.mean(scores), np.std(scores)))
# print(gs.best_params_)
# # KNN
# pipe_knn = Pipeline([['pca', PCA()], ['knn', KNeighborsClassifier()]])
# param_grid = [{'pca__n_components': list(range(1, X_train.shape[1] + 1)), 'knn__n_neighbors': list(range(1, 8))}]
# gs = GridSearchCV(estimator=pipe_knn,
#                   param_grid=param_grid,
#                   cv=2,
#                   scoring='accuracy',
#                   n_jobs=-1)
# gs.fit(X_train, y_train)
# scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=5)
# print('CV 2x5 accuracy: %.2f +- %.2f' % (np.mean(scores), np.std(scores)))
# print(gs.best_params_)
########################################################################################################################
# build model (if haven't chosen)                                                                                      #
########################################################################################################################
# pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
########################################################################################################################
# brute_force (adjust parameters)                                                                                      #
########################################################################################################################
# # SVC
# pipe_svc = Pipeline([['sc', StandardScaler()], ['pca', PCA()], ['svc', SVC(random_state=1)]])
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# param_grid = [
#     {'svc__C': param_range, 'svc__kernel': ['linear'], 'pca__n_components': list(range(1, X_train.shape[1] + 1))},
#     {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf'],
#      'pca__n_components': list(range(1, X_train.shape[1] + 1))}]
# gs = GridSearchCV(estimator=pipe_svc,
#                   param_grid=param_grid,
#                   cv=10,
#                   scoring='accuracy',
#                   n_jobs=-1)
# gs.fit(X_train, y_train)
# best_model = gs.best_estimator_
# print(gs.best_params_)
# print('model_1 accuracy: %.2f' % gs.best_score_)
# # RF
# pipe_rf = Pipeline([['pca', PCA()], ['rf', RandomForestClassifier(random_state=1)]])
# param_grid = [{'rf__n_estimators': list(range(5, 15)),
#                'rf__criterion': ['gini', 'entropy'],
#                'rf__max_depth': list(range(5, 10)),
#                'pca__n_components': list(range(1, X_train.shape[1] + 1))}]
# gs = GridSearchCV(estimator=pipe_rf,
#                   param_grid=param_grid,
#                   cv=10,
#                   scoring='accuracy',
#                   n_jobs=-1)
# gs.fit(X_train, y_train)
# best_model_2 = gs.best_estimator_
# print(gs.best_params_)
# print('model_2 accuracy: %.2f' % gs.best_score_)
# # LR
# pipe_lr = Pipeline([['sc', StandardScaler()], ['pca', PCA()], ['lr', LogisticRegression(random_state=1)]])
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# param_grid = [{'lr__C': param_range, 'pca__n_components': list(range(1, X_train.shape[1] + 1))}]
# gs = GridSearchCV(estimator=pipe_lr,
#                   param_grid=param_grid,
#                   cv=10,
#                   scoring='accuracy',
#                   n_jobs=-1)
# gs.fit(X_train, y_train)
# best_model_3 = gs.best_estimator_
# print(gs.best_params_)
# print('model_3 accuracy: %.2f' % gs.best_score_)
# # KNN
# pipe_knn = Pipeline([['pca', PCA()], ['knn', KNeighborsClassifier()]])
# param_grid = [{'pca__n_components': list(range(1, X_train.shape[1] + 1)), 'knn__n_neighbors': list(range(1, 8))}]
# gs = GridSearchCV(estimator=pipe_knn,
#                   param_grid=param_grid,
#                   cv=10,
#                   scoring='accuracy',
#                   n_jobs=-1)
# gs.fit(X_train, y_train)
# best_model_4 = gs.best_estimator_
# print(gs.best_params_)
# print('model_4 accuracy: %.2f' % gs.best_score_)
########################################################################################################################
# ensemble                                                                                                             #
########################################################################################################################
# clf_1 = Pipeline([['sc', StandardScaler()], ['pca', PCA(6)], ['svc', SVC(random_state=1, C=100.0, gamma=0.1,
#                                                                          kernel='rbf')]])
# clf_2 = Pipeline([['pca', PCA(6)], ['rf', RandomForestClassifier(random_state=1, criterion='entropy', max_depth=9,
#                                                                  n_estimators=12)]])
# clf_3 = pipe_knn = Pipeline([['pca', PCA(6)], ['knn', KNeighborsClassifier(n_neighbors=5)]])
# weights = [0.99, 0.88, 0.94]
# clf_e = VotingClassifier(estimators=[('svc', clf_1), ('rf', clf_2), ('knn', clf_3)], voting='hard', weights=weights)
#
# best_score = 0.0
# best_model = None
# for clf, label in zip([clf_1, clf_2, clf_3, clf_e], ['SVC', 'RF', 'KNN', 'Ensemble']):
#     scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=10)
#     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#     if scores.mean() >= best_score:
#         best_score = scores.mean()
#         best_model = clf
# pickle_model.save_model(r'C:\Users\ASUS ROG\PycharmProjects\Machine '
#                         r'Learning\UCI_Car_Evaluation\models\UCI_Car_Evaluation_1-1.pickle', best_model)
# # result:
# # Accuracy: 0.99 (+/- 0.01) [SVC]
# # Accuracy: 0.88 (+/- 0.04) [RF]
# # Accuracy: 0.94 (+/- 0.02) [KNN]
# # Accuracy: 0.96 (+/- 0.02) [Ensemble]
########################################################################################################################
# fit on whole train_data and scoring on test_data                                                                     #
########################################################################################################################
# model = pickle_model.load_model(r'C:\Users\ASUS ROG\PycharmProjects\Machine '
#                                 r'Learning\UCI_Car_Evaluation\models\UCI_Car_Evaluation_1-1.pickle')
# model.fit(X_train, y_train)
# acc = model.score(X_test, y_test)
# print('Test accuracy: %.4f' % acc)

# Test accuracy: 0.9942
