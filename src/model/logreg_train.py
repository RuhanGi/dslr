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
		# courses =['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Ancient Runes']
		# TODO check if linear variables affect anything
		courses =['Herbology', 'Defense Against the Dark Arts', 'Ancient Runes']
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

def epoch(ndata, y, th):
	learningRate = 0.05
	m = len(ndata)
	grad = np.zeros_like(th)
	probs = softmax(ndata, th)

	classes = list(th.index)
	for i in range(m):
		for c in range(len(classes)):
			grad[c, :] += (probs[i, c] - (1 if y[i] == classes[c] else 0)) * ndata[i]
	return th - grad / m

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
		maxiterations = 10
		tolerance = 10**-12
		
		for i in range(maxiterations):
			prvth = th.copy()
			th = epoch(ndata, y, th)
			if i % 1000 == 0 and np.all(np.abs(th.values - prvth.values) < tolerance):
				break
		return th
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

		# TODO BONUS implement a stochastic gradient descent
		# TODO other optimization aglorithms:
		# TODO - Batch GD/mini-batch GD
		# TODO - adagrad/adadelta/adam
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