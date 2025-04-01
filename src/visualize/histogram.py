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
	courses = df.select_dtypes(include=['float64']).columns
	houses = df['Hogwarts House'].unique()
	colors = {house: color for house, color in zip(houses, sns.color_palette("husl", len(houses)))}
	
	num_courses = len(courses)
	rows = 3
	cols = (num_courses // rows) + (num_courses % rows > 0)
	fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
	axes = axes.flatten()
	
	for i, course in enumerate(courses):
		ax = axes[i]
		for house in houses:
			sns.histplot(df[df['Hogwarts House'] == house][course], 
						 bins=15, kde=True, color=colors[house], 
						 label=house, ax=ax, alpha=0.6)
		ax.set_title(course)
		ax.legend()
	
	for j in range(i + 1, len(axes)):
		fig.delaxes(axes[j])
	
	plt.tight_layout()
	plt.gcf().canvas.mpl_connect('key_press_event', lambda event: plt.close() if event.key == 'escape' else None)
	plt.show()

def main():
	if len(sys.argv) != 2:
		print(RED + "Pass Data to Describe!" + RESET)
		sys.exit(1)
	df = loadData(sys.argv[1])
	df = cleanData(df)
	plotData(df)
	
if __name__ == "__main__":
	main()