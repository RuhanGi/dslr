import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import sys

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[97m"
BLACK = "\033[98m"
RESET = "\033[0m"

def loadData(fil):
	try:
		return pd.read_csv(fil)
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def cleanData(df):
	try:
		houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
		df = df[df['Hogwarts House'].isin(houses)]
		float_cols = df.select_dtypes(include=['float64'])
		df.dropna(inplace=True, subset=float_cols.columns)
		df.drop_duplicates(inplace = True)
		
		# TODO check if linear variables affect anything
		# courses =['Herbology', 'Defense Against the Dark Arts', 'Ancient Runes']
		courses =['Herbology', 'Defense Against the Dark Arts', 'Ancient Runes', 'Charms']
		df = df[courses + ['Hogwarts House']]
		return df
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def softmax(ndata, th):
	z = np.dot(ndata, th.T)
	exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True)) 
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	return probs

# * Mini-Batch GD
def minibatchEpoch(ndata, y, th):
	learningRate = 0.3
	batch_size = 32
	m = len(ndata)
	num_batches = (m + batch_size-1) // batch_size
	classes = list(th.index)

	for i in range(num_batches):
		start = i * batch_size
		end = min((i + 1) * batch_size, m)
		X_batch = ndata[start:end]
		y_batch = y[start:end]
		probs = softmax(X_batch, th)
		y_onehot = np.zeros_like(probs)
		for i, label in enumerate(y_batch):
			y_onehot[i, classes.index(label)] = 1
		grad = np.dot((probs - y_onehot).T, X_batch)
		th = th - learningRate * grad / (end-start)
	return th

# * Batch GD
def batchEpoch(ndata, y, th):
	learningRate = 0.3
	m = len(ndata)
	probs = softmax(ndata, th)

	classes = list(th.index)
	y_onehot = np.zeros_like(probs)
	for i, label in enumerate(y):
		y_onehot[i, classes.index(label)] = 1
	grad = np.dot((probs - y_onehot).T, ndata)
	return th - learningRate * grad / m

# * Stochastic GD
def stochEpoch(ndata, y, th):
	learningRate = 0.3
	classes = list(th.index)
	m = len(ndata)

	for i in range(m):
		x_i = ndata[i].reshape(1, -1)
		probs = softmax(x_i, th)
		y_onehot = np.zeros_like(probs)
		y_onehot[0, classes.index(y[i])] = 1
		th = th - learningRate * np.dot((probs - y_onehot).T, x_i)
	return th

def trainModel(data, y, headers, n):
	try:
		classes = np.unique(y)
		th = pd.DataFrame(np.zeros((len(classes), n)), columns=['Bias'] + list(headers), index=classes)
		
		mins = np.min(data, axis=0)
		maxs = np.max(data, axis=0)
		ranges = maxs - mins
		ranges[ranges == 0] = 1
		ndata = (data - mins) / ranges
		ndata = np.hstack((np.ones((ndata.shape[0], 1)), ndata))

		maxiterations = 10000
		tolerance = 10**-3
		for i in range(maxiterations):
			prvth = th.copy()
			th = minibatchEpoch(ndata, y, th)
			print(f"\rEpoch [{i}/{maxiterations}]",end="")
			if i % 100 == 0 and np.all(np.abs(th.values - prvth.values) < tolerance):
				print(f"\rEpoch [{i}/{maxiterations}]")
				break
		print(GREEN + "\rModel Trained!" + (" " * 30) + RESET)
		
		# TODO - adagrad/adadelta/adam
		
		for i in range(th.shape[0]):
			weights = th.iloc[i, 1:].values
			denorm_weights = weights / ranges
			denorm_bias = th.iloc[i, 0] - np.sum(weights * mins / ranges)
			th.iloc[i, 1:] = denorm_weights
			th.iloc[i, 0] = denorm_bias

		return th
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def main():
	if len(sys.argv) != 2:
		print(RED + "Pass Training Data!" + RESET)
		sys.exit(1)
	
	df = loadData(sys.argv[1])
	df = cleanData(df)
	data, y = df.iloc[:, :-1].to_numpy(), df.iloc[:,-1].to_numpy()
	th = trainModel(data, y, df.columns[:-1].to_numpy(), df.shape[1])
	print(th)
	th.to_csv("thetas.csv")

if __name__ == "__main__":
	main()