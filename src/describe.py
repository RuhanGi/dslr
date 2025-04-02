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
		float_cols = df.select_dtypes(include=['float64'])
		df.dropna(inplace=True, subset=float_cols.columns)
		df.drop_duplicates(inplace = True)
		return df
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

# * wrong format integer among strings
# ? df['Birthday'] = pd.to_datetime(df['Birthday'])
# * Remove rows with NaN only in floats so far
# * wrong data?
# ? try removing MAJOR outliers

def count(column):
	c = 0
	for i in column:
		c += 1
	return c

def mean(column):
	c = 0
	sum = 0
	for a in column:
		c += 1
		sum += a
	return c, sum / c

def std(column, c, m):
	sum = 0
	for a in column:
		sum += (a - m)**2
	return (sum / (c-1))**(1/2)

def quartiles(column, c):
	column = column.copy().sort_values()
	r,p = math.modf(1 * (c+1)/4)
	p = int(p)
	q1 = column.iloc[p] + r * (column.iloc[p+1] - column.iloc[p])
	r,p = math.modf(2 * (c+1)/4)
	p = int(p)
	q2 = column.iloc[p] + r * (column.iloc[p+1] - column.iloc[p])
	r,p = math.modf(3 * (c+1)/4)
	p = int(p)
	q3 = column.iloc[p] + r * (column.iloc[p+1] - column.iloc[p])
	return column.iloc[0], q1, q2, q3, column.iloc[-1]

def extractInfo(df):
	try:
		row_headers = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
		column_headers = df.select_dtypes(include=['float64']).columns
		stats = pd.DataFrame(df, index=row_headers, columns=column_headers)
		
		for col in stats.columns:
			stats.loc["Count", col], stats.loc["Mean", col] = mean(df[col])
			stats.loc["Std", col] = std(df[col], stats.loc["Count", col], stats.loc["Mean", col])
			stats.loc["Min", col], stats.loc["25%", col], stats.loc["50%", col], stats.loc["75%", col], \
				  stats.loc["Max", col] = quartiles(df[col], stats.loc["Count", col])

		# TODO BONUS add more fields
		# stats.rename(columns={'Defense Against the Dark Arts': 'Dark Arts'}, inplace=True)
		# stats.rename(columns={'Care of Magical Creatures': 'Magical Creatures'}, inplace=True)
		return stats
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def main():
	if len(sys.argv) != 2:
		print(RED + "Pass Data to Describe!" + RESET)
		sys.exit(1)
	
	df = loadData(sys.argv[1])
	# print(YELLOW)
	# print(df.info())

	df = cleanData(df)
	# print(GREEN)
	print(df.info())

	# pd.options.display.float_format = '{:.6f}'.format
	stats = extractInfo(df)
	print(stats)

if __name__ == "__main__":
	main()