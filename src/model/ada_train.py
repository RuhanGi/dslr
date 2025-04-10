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


# TODO - adagrad/adadelta/adam

def adagradEpoch(ndata, y, th, cache):
	m = len(ndata)
	probs = softmax(ndata, th)

	classes = list(th.index)
	y_onehot = np.zeros_like(probs)
	for i, label in enumerate(y):
		y_onehot[i, classes.index(label)] = 1
	grad = np.dot((probs - y_onehot).T, ndata) / m

	cache += grad**2
	epsilon = 10**-8
	learningRate = 0.3 / (np.sqrt(cache) + epsilon)

	return th - learningRate * grad, cache

def rmspropEpoch(ndata, y, th, cache):
	m = len(ndata)
	probs = softmax(ndata, th)

	classes = list(th.index)
	y_onehot = np.zeros_like(probs)
	for i, label in enumerate(y):
		y_onehot[i, classes.index(label)] = 1
	grad = np.dot((probs - y_onehot).T, ndata) / m

	decay = 0.95
	cache = cache * decay + (1-decay) * grad**2
	epsilon = 10**-8
	learningRate = 0.05 / (np.sqrt(cache) + epsilon)

	return th - learningRate * grad, cache

def adamEpoch(ndata, y, th, momentum, velocity, t):
	m = len(ndata)
	probs = softmax(ndata, th)

	classes = list(th.index)
	y_onehot = np.zeros_like(probs)
	for i, label in enumerate(y):
		y_onehot[i, classes.index(label)] = 1
	grad = np.dot((probs - y_onehot).T, ndata) / m

	decay1, decay2 = 0.9, 0.99
	momentum = decay1 * momentum + (1-decay1) * grad
	velocity = decay2 * velocity + (1-decay2) * grad**2
	epsilon = 10**-8
	learningRate = 0.001 / (np.sqrt(velocity / (1-decay2**t)) + epsilon)

	return th - learningRate * (momentum / (1-decay1**t)), momentum, velocity


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
		tolerance = 10**-4
		momentum = np.zeros_like(th.values)
		velocity = np.zeros_like(th.values)
		for t in range(1, maxiterations+1):
			prvth = th.copy()
			# th, velocity = rmspropEpoch(ndata, y, th, velocity)
			th, momentum, velocity = adamEpoch(ndata, y, th, momentum, velocity, t)
			maxDiff = np.max(np.abs(th.values-prvth.values))
			print(f"\rEpoch [{t}/{maxiterations}]: {maxDiff:.6f}",end="")
			if maxDiff < tolerance:
				print(f"\rEpoch [{t}/{maxiterations}]")
				break
		print(GREEN + "\rModel Trained!" + (" " * 30) + RESET)
		
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