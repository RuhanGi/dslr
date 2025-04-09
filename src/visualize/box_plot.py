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
		df.set_index("Index", inplace=True)
		houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
		df = df[df['Hogwarts House'].isin(houses)]
		float_cols = df.select_dtypes(include=['float64'])
		df.dropna(inplace=True, subset=float_cols.columns)
		df.drop_duplicates(inplace = True)
		
		df.rename(columns={'Defense Against the Dark Arts': 'Dark Arts'}, inplace=True)
		df.rename(columns={'Care of Magical Creatures': 'Creatures'}, inplace=True)
		df.rename(columns={'History of Magic': 'History'}, inplace=True)
		return df
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def plotData(df):
	score_cols = df.select_dtypes(include="float64").columns
	df_long = df.melt(id_vars="Hogwarts House", value_vars=score_cols,
					  var_name="Course", value_name="Score")
	plt.figure(figsize=(20, 12))
	sns.boxplot(data=df_long, x="Course", y="Score", hue="Hogwarts House", log_scale=True)
	plt.xticks(rotation=45)
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