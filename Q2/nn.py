import numpy as np
import sys
from Q2a import *

output_file = sys.argv[4]
batchSize = int(sys.argv[5])
hidden_layer_list = [int(i) for i in sys.argv[6].split()]
activation_type = sys.argv[7]

x_train = np.load(sys.argv[1])
y_train = np.load(sys.argv[2])
x_test = np.load(sys.argv[3])
#y_test = np.load('../kannada_digits/neural_network_kannada/y_test.npy')

x_train = np.array([x_train[i].flatten() for i in range(x_train.shape[0])])
x_test = np.array([x_test[i].flatten() for i in range(x_test.shape[0])])
x_train = x_train/255
x_test = x_test/255
y_train = ohe(y_train)
#y_test = ohe(y_test)

if activation_type == 'softmax':
	nn = NeuralNet(x_train.shape[1], y_train.shape[1], hidden_layer_list, 0.5, batchSize, 'adaptive', sigmoid)
else:
	nn = NeuralNet(x_train.shape[1], y_train.shape[1], hidden_layer_list, 0.5, batchSize, 'adaptive', relu)

start = time.time()
nn.fit(x_train, y_train)
end = time.time()
train_pred = nn.predict(x_train)
train_acc = nn.accuracy(y_train, train_pred)
test_pred = nn.predict(x_test)
#test_acc = nn.accuracy(y_test, test_pred)

f = open(output_file, 'w')
for pred in test_pred:
	label = np.argmax(pred)
	print(label, file=f)

#print("Activation type: ", activation_type)
#print("Batch Size: ", batchSize)
#print("Hidden Layer List: ", hidden_layer_list)
#print("Training Time = ", (end-start))
#print("Training Accuracy = ", train_acc)
#print("Test Accuracy = ", test_acc)