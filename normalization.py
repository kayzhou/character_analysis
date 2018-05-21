# -*- coding: utf-8 -*-
__author__ = 'Kay'

import numpy as np
from sklearn import preprocessing


def scaler(in_name, out_name):
    raw_data = np.loadtxt(in_name)
    X = raw_data[:, 1:]

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)

    # scaler = preprocessing.StandardScaler()
    # X_scaled = scaler.fit_transform(X)

    out_file = open(out_name, "w")
    for i, line in enumerate(X_scaled):
        out_file.write(str(raw_data[i, 0])[:10] + " " + " ".join([str(x) for x in line]) + '\n')


def scaler_fit(in_name, x_name, out_name):
    raw_data = np.loadtxt(in_name)
    source_X = raw_data[:, 1:]

    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_scaler.fit(source_X)

    target_data = np.loadtxt(x_name)
    target_X = target_data[:, 1:]
    X_scaled = min_max_scaler.fit_transform(target_X)

    out_file = open(out_name, "w")
    for i, line in enumerate(X_scaled):
        out_file.write(str(target_data[i, 0])[:10] + " " + " ".join([str(x) for x in line]) + '\n')


if __name__ == '__main__':
    scaler("data/features/train_IGNORE_404.txt", "data/features/train_IGNORE_404_NOR.txt")
    # scaler_fit("data/features/train_IGNORE_404.txt", "data/features/large_IGNORE_404.txt", "data/features/large_IGNORE_404_NOR.txt")