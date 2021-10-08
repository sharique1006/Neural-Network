import numpy as np
import matplotlib.pyplot as plt
import time
import sys

def ohe(y):
	labels = np.unique(y)
	ohy = np.zeros((y.shape[0], len(labels)))
	ohy[np.arange(y.shape[0]), y] = 1
	return ohy

num_layers = 0

def sigmoid(z):
	return 1.0/(1.0 + np.exp(-z))

def relu(z):
	z[z <= 0] = 0
	return z

def relu_derivative(z):
	z[z <= 0] = 0
	z[z > 0] = 1
	return z

class Layer():
	def __init__(self, in_size, out_size):
		global num_layers
		self.in_size = in_size
		self.out_size = out_size
		if in_size != -1:
			self.w = np.random.normal(0, 0.1, in_size*out_size).reshape(in_size, out_size)
		num_layers += 1
		self.delta = None

class NeuralNet():
	def __init__(self, in_size, r, layers, eta, batchSize, learning='static', activation=sigmoid):
		global num_layers
		num_layers = 0
		self.eta = eta
		self.batchSize = batchSize
		self.activation = activation
		self.learning = learning
		self.layers = []
		self.layers.append(Layer(-1, in_size+1))
		for i in range(len(layers)):
			self.layers.append(Layer(self.layers[-1].out_size, layers[i]))
		self.layers.append(Layer(self.layers[-1].out_size, r))

	def forward(self, x):
		global num_layers
		self.layers[0].a = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
		for i in range(1, num_layers-1):
			self.layers[i].a = self.activation(np.dot(self.layers[i-1].a, self.layers[i].w))
		self.layers[-1].a = sigmoid(np.dot(self.layers[-2].a, self.layers[-1].w))

	def backward(self, y):
		self.layers[-1].delta = (1/float(y.shape[0]))*(y - self.layers[-1].a) * self.layers[-1].a * (1 - self.layers[-1].a)
		for i in reversed(range(1, num_layers-1)):
			if self.activation == sigmoid:
				self.layers[i].delta = np.dot(self.layers[i+1].delta, self.layers[i+1].w.T) * self.layers[i].a * (1 - self.layers[i].a)
			else:
				self.layers[i].delta = np.dot(self.layers[i+1].delta, self.layers[i+1].w.T) * relu_derivative(self.layers[i].a)

	def fit(self, x, y):
		r = self.batchSize
		num_batches = int(x.shape[0]/r)
		converged = False
		epochs = 0
		prevCost = 0
		while not converged:
			epochs += 1
			for i in range(num_batches):
				x_batch = x[i*r:(i+1)*r]
				y_batch = y[i*r:(i+1)*r]
				self.forward(x_batch)
				self.backward(y_batch)
				for j in range(1, num_layers):
					if self.learning == 'static':
						self.layers[j].w += self.eta * np.dot(self.layers[j-1].a.T, self.layers[j].delta)
					else:
						self.layers[j].w += (self.eta/np.sqrt(epochs)) * np.dot(self.layers[j-1].a.T, self.layers[j].delta)
				cost = 0.5/float(r)*((y_batch - self.layers[-1].a)**2).sum()
			error = abs(prevCost - cost)
			prevCost = cost
			if (epochs > 10 and error < 1e-4) or epochs > 100:
				converged = True
		#print("Epochs = ", epochs, "Final Cost = ", cost)

	def predict(self, x):
		self.forward(x)
		prediction = self.layers[-1].a
		prediction = (prediction == prediction.max(axis=1)[:,None]).astype(int)
		return prediction

	def accuracy(self, y, prediction):
		accuracy = 0
		for i in range(len(prediction)):
			if (prediction[i] == y[i]).all():
				accuracy += 1
		accuracy = accuracy/y.shape[0]
		return accuracy

def PlotAccuracyvsHLU(x_train, y_train, x_test, y_test, eta, learning, activation, file1, file2):
	print("Learning Mode: ", learning)
	HLU = [1, 10, 50, 100, 500]
	train_accuracies = []
	test_accuracies = []
	training_time = []
	for u in HLU:
		print("Number of Units in Hidden Layer = ", u)
		nn = NeuralNet(x_train.shape[1], y_train.shape[1], [u], eta, 100, learning=learning, activation=activation)
		start = time.time()
		nn.fit(x_train, y_train)
		end = time.time()
		train_time = (end-start)
		train_pred = nn.predict(x_train)
		train_acc = nn.accuracy(y_train, train_pred)
		test_pred = nn.predict(x_test)
		test_acc = nn.accuracy(y_test, test_pred)
		print("Training Time = ", train_time)
		print("Training Accuracy = ", train_acc)
		print("Test Accuracy = ", test_acc)
		training_time.append((end-start))
		train_accuracies.append(train_acc)
		test_accuracies.append(test_acc)

	print("HLU = ", np.array(HLU))
	print("Training Times = ", np.array(training_time))
	print("Train Accuracies = ", np.array(train_accuracies))
	print("Test Accuracy = ", np.array(test_accuracies))
	plt.figure()
	plt.plot(HLU, train_accuracies, label='Train Accuracy')
	plt.plot(HLU, test_accuracies, label='Test Accuracy')
	plt.xlabel("Number of Hidden Layer Units")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs Number of HLU in a single hidden layer Neural Network")
	plt.legend()
	plt.savefig(file1)
	plt.show()
	plt.close()

	plt.figure()
	plt.plot(HLU, training_time)
	plt.title("Training Time vs Number of HLU")
	plt.xlabel("Number of Hidden Layer Units")
	plt.ylabel("Training Time")
	plt.savefig(file2)
	plt.show()
	plt.close()