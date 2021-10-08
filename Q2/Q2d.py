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

print("Sigmoid Model")
nn_sigmoid = NeuralNet(x_train.shape[1], y_train.shape[1], [100, 100], 0.5, 100, 'adaptive', sigmoid)
start = time.time()
nn_sigmoid.fit(x_train, y_train)
end = time.time()
train_pred_sigmoid = nn_sigmoid.predict(x_train)
train_acc_sigmoid = nn_sigmoid.accuracy(y_train, train_pred_sigmoid)
test_pred_sigmoid = nn_sigmoid.predict(x_test)
test_acc_sigmoid = nn.accuracy(y_test, test_pred_sigmoid)
print("Training Time = ", (end-start))
print("Training Accuracy = ", train_acc_sigmoid)
print("Test Accuracy = ", test_acc_sigmoid)
print()
print("Relu Model")
nn_relu = NeuralNet(x_train.shape[1], y_train.shape[1], [100, 100], 0.5, 100, 'adaptive', relu)
start = time.time()
nn_relu.fit(x_train, y_train)
end = time.time()
train_pred_relu = nn_relu.predict(x_train)
train_acc_relu = nn_relu.accuracy(y_train, train_pred_relu)
test_pred_relu = nn_relu.predict(x_test)
test_acc_relu = nn_relu.accuracy(y_test, test_pred_relu)
print("Training Time = ", (end-start))
print("Training Accuracy = ", train_acc_relu)
print("Test Accuracy = ", test_acc_relu)