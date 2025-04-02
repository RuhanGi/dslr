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
		courses =['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Ancient Runes']
		df = df[courses + ['Hogwarts House']]
		return df
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def epoch(ndata, act, th):
	learningRate = 0.05
	m = len(ndata)
	diff = (th[0] + np.dot(ndata, th[1:])) - act
	th[0] -= learningRate / m * np.sum(diff)
	th[1:] -= learningRate / m * (diff @ ndata)
	return th

def trainModel(data, n):
	try:
		th = np.zeros(n)
		mins = np.min(data, axis=0)
		maxs = np.max(data, axis=0)
		ranges = maxs - mins
		ranges[ranges == 0] = 1
		ndata = (data - mins) / ranges
		ndata, act = ndata[:, :-1], ndata[:, -1]
		maxiterations = 1000000
		tolerance = 10**-12
		
		for i in range(maxiterations):
			prvth = th.copy()
			th = epoch(ndata, act, th)
			if i % 1000 == 0 and np.all(np.abs(th - prvth) < tolerance):
				break

		for i in range(n-1):
			th[i+1] *= ranges[n-1] / ranges[i]
		th[0] = mins[-1] + th[0] * ranges[-1] - np.sum(th[1:] * mins[:-1])
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
	print(df.info())
	print(df.columns)
	# th = trainModel(df, len(df))
	# np.save("thetas.npy", {"theta": th, "headers": df.columns})

if __name__ == "__main__":
	main()