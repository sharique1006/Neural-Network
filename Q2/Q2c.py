import numpy as np
import sys
from Q2a import *

x_train = np.load('../kannada_digits/neural_network_kannada/X_train.npy')
y_train = np.load('../kannada_digits/neural_network_kannada/y_train.npy')
x_test = np.load('../kannada_digits/neural_network_kannada/X_test.npy')
y_test = np.load('../kannada_digits/neural_network_kannada/y_test.npy')

x_train = np.array([x_train[i].flatten() for i in range(x_train.shape[0])])
x_test = np.array([x_test[i].flatten() for i in range(x_test.shape[0])])
x_train = x_train/255
x_test = x_test/255
y_train = ohe(y_train)
y_test = ohe(y_test)

PlotAccuracyvsHLU(x_train, y_train, x_test, y_test, 0.5, 'adaptive', sigmoid, 'Q2bacc.png', 'Q2btime.png')