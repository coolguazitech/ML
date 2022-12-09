import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn import model_selection

##############################
# plot a graph of a function #
##############################

# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))
# z = np.arange(-7, 7, 0.1)
# phi_z = sigmoid(z)
# plt.plot(z, phi_z)
# plt.axvline(0.0, color='k')
# plt.ylim(-0.1, 1.1)
# plt.xlabel('z')
# plt.ylabel('$\phi(z)$')
# plt.yticks([0.0, 0.5, 1.0])
# plt.show()

###################################
# make a file-like csv data in IO #
###################################

# csv_data = \
#     '''
# A,B,C,D
# 1.0,2.0,3.0,4.0
# 5.0,6.0,,8.0
# 9.0,10.0,11.0
# '''
# df = pd.read_csv(StringIO(csv_data))

#################
# preprocessing #
#################

# orderable feature:
# map(mapping_dict)

# non orderable feature:
# LabelEncoder()

# make example_data
# df = pd.DataFrame([['green', 'M', 10.1, 'class1'], ['red', 'L', 13.5, 'class2'], ['blue', 'XL', 15.3, 'class1']])
# df.columns = ['color', 'size', 'price', 'classlabel']

# size_mapping = dict([('M',1),('L',2),('XL',3)])
# inv_size_mapping = dict([(v, k) for k, v in size_mapping.items()])

# df['size'] = df['size'].map(size_mapping)
# # df['size'] = df['size'].map(inv_size_mapping)

# le = LabelEncoder()
# df['color'] = le.fit_transform(df['color'])
# df['classlabel'] = le.fit_transform(df['classlabel'])
# df['classlabel'] = le.inverse_transform(df['classlabel'])

# ohe = OneHotEncoder()
# ohe.fit_transform(df[['color', 'classlabel']]).toarray()

# pd.get_dummies(df)

# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)

#####################
# feature_selection #
#####################

# see my_utils

###############
# train steps #
###############

# set data: X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
# choose model
# build model: nested-cross-validation
# preparation:  (1) cross validation to see if the hyper-parameters are good
#               (2) monitor to see why or why not hyper-parameters are good
#               (3) hyper-parameters adjustment (better)
# fit on whole train_data and scoring on test_data

###################
# model_selection #
###################

# choose model
# choose model: nested-cross-validation

# hyper-parameters adjustment
# brute_force: GridSearchCV
# monitor