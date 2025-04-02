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

def epoch(ndata, classes, th):
	learningRate = 0.05
	m = len(ndata)
	z = th[0] + np.dot(ndata, th[1:])
	h = 1 / (1 + np.exp(z))
	th[0] -= learningRate / m * np.sum(h - classes)
	th[1:] -= learningRate / m * (diff @ ndata)
	return th

def trainModel(data, headers, n):
	try:
		houses = np.unique(headers)
		numclasses = len(houses)
		th = np.zeros((n, numclasses))
		mins = np.min(data, axis=0)
		maxs = np.max(data, axis=0)
		ranges = maxs - mins
		ranges[ranges == 0] = 1
		ndata = (data - mins) / ranges
		maxiterations = 1000000
		tolerance = 10**-12
		
		# for i in range(maxiterations):
		# 	prvth = th.copy()
		# 	th = epoch(ndata, headers, th)
		# 	if i % 1000 == 0 and np.all(np.abs(th - prvth) < tolerance):
		# 		break

		# ? USE SOFTMAX FOR DETECTION
		
		# for i in range(n-1):
		# 	th[i+1] *= ranges[n-1] / ranges[i]
		# th[0] = mins[-1] + th[0] * ranges[-1] - np.sum(th[1:] * mins[:-1])
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
	data, headers = df.iloc[:, :-1].to_numpy(), df.iloc[:,-1].to_numpy()
	th = trainModel(data, headers, df.shape[1])
	print(th)
	# np.save("thetas.npy", {"theta": th, "headers": headers})

if __name__ == "__main__":
	main()