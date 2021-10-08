import numpy as np
import sys
from Q2a import *
from sklearn.neural_network import MLPClassifier

x_train = np.load('../kannada_digits/neural_network_kannada/X_train.npy')
y_train = np.load('../kannada_digits/neural_network_kannada/y_train.npy')
x_test = np.load('../kannada_digits/neural_network_kannada/X_test.npy')
y_test = np.load('../kannada_digits/neural_network_kannada/y_test.npy')

x_train = np.array([x_train[i].flatten() for i in range(x_train.shape[0])])
x_test = np.array([x_test[i].flatten() for i in range(x_test.shape[0])])
x_train = x_train/255
x_test = x_test/255

clf = MLPClassifier(hidden_layer_sizes=(100,100), activation='relu', solver='sgd', batch_size=100, learning_rate='invscaling', learning_rate_init=0.5, power_t=0.5, max_iter=100, verbose=True)
start = time.time()
clf.fit(x_train, y_train)
end = time.time()
print("Training Time = ", (end-start))
train_acc = clf.score(x_train, y_train)
test_acc = clf.score(x_test, y_test)
print("Training Accuracy = ", train_acc)
print("Test Accuracy = ", test_acc)
