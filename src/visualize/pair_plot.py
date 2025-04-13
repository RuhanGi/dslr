import matplotlib.pyplot as plt
import seaborn as sns
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
		return df
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def plotData(df):
	courses =['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Charms', 'Flying']
	
	df_subset = df[courses + ['Hogwarts House']]
	fig1 = sns.pairplot(df_subset, hue="Hogwarts House")
	fig1.fig.canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
	plt.show()
	plt.close(fig1.fig)

	df_float = df.select_dtypes(include='float64').copy()
	df_float['Hogwarts House'] = df['Hogwarts House']
	fig2 = sns.pairplot(df_float, hue="Hogwarts House")
	fig2.fig.savefig("pairplot.png", dpi=300)
	plt.close(fig2.fig)

def main():
	if len(sys.argv) != 2:
		print(RED + "Pass Data to Describe!" + RESET)
		sys.exit(1)
	df = loadData(sys.argv[1])
	df = cleanData(df)
	plotData(df)
	
if __name__ == "__main__":
	main()