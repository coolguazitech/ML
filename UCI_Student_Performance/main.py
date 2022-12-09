import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plot
import pickle
import os
from matplotlib import style

dir_ = 'models/'
path = os.path.join(dir_, 'UCI_Student_Performance_lr.pickle')

data = pd.read_csv('student-mat.csv', sep=';')
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
# print(data.head())

predict = 'G3'

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, stratify=y)

# best = 0.0
# for _ in range(100):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#
#     lr = linear_model.LinearRegression()
#     lr.fit(x_train, y_train)
#
#     acc = lr.score(x_test, y_test)
#     print(acc)
#
#     if acc > best:
#         best = acc
#         if not os.path.isdir(dir_):
#             os.makedirs(dir_)
#         with open(path, 'wb') as f:
#             pickle.dump(lr, f)

pickle_in = open(path, "rb")
lr = pickle.load(pickle_in)

acc = lr.score(x_test, y_test)
print(acc)

print('Coefficient: ', lr.coef_)
print('Intercept: ', lr.intercept_)

predictions = lr.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])

p = "G1"
style.use("ggplot")
plot.scatter(data[p], data["G3"])
plot.xlabel(p)
plot.ylabel("Final Grade")
plot.show()
