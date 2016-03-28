# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
from sklearn import preprocessing

in_name = "data/features/315_features.txt"
out_name = "data/features/315_features_standard.txt"
raw_data = np.loadtxt(in_name)
X = raw_data[:, 1:]

# min_max_scaler = preprocessing.MinMaxScaler()
# X_scaled = min_max_scaler.fit_transform(X)

scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

out_file = open(out_name, "w")
for line in X_scaled:
    out_file.write(" ".join([str(x) for x in line]) + '\n')