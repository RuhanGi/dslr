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

def skew(column, m, c, s):
	sums = 0
	sumk = 0
	for a in column:
		sums += ((a - m)/s)**3
		sumk += ((a - m)/s)**4
	return sums / c, sumk / c

def countOutliers(column, q1, q3):
	lowest = q1 - 1.5 * (q3 - q1)
	highest = q3 + 1.5 * (q3 - q1)
	outlies = 0
	for a in column:
		if (a < lowest or a > highest):
			outlies += 1
	return outlies

def extractInfo(df):
	try:
		row_headers = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max",
				  "IQR", "Range", "Skewness", "Kurtosis", "#Outliers"]
		column_headers = df.select_dtypes(include=['float64']).columns
		stats = pd.DataFrame(df, index=row_headers, columns=column_headers)

		for col in stats.columns:
			stats.loc["Count", col], stats.loc["Mean", col] = mean(df[col])
			stats.loc["Std", col] = std(df[col], stats.loc["Count", col], stats.loc["Mean", col])
			stats.loc["Min", col], stats.loc["25%", col], stats.loc["50%", col], stats.loc["75%", col], \
					stats.loc["Max", col] = quartiles(df[col], stats.loc["Count", col])
			stats.loc["IQR", col] = stats.loc["75%", col] - stats.loc["25%", col]
			stats.loc["Range", col] = stats.loc["Max", col] - stats.loc["Min", col]
			stats.loc["Skewness", col], stats.loc["Kurtosis", col] = skew(df[col], \
					stats.loc["Mean", col], stats.loc["Std", col], stats.loc["Count", col])
			stats.loc["#Outliers", col] = countOutliers(df[col], stats.loc["25%", col], stats.loc["75%", col])

		stats.rename(columns={'Defense Against the Dark Arts': 'Dark Arts'}, inplace=True)
		stats.rename(columns={'Care of Magical Creatures': 'Creatures'}, inplace=True)
		stats.rename(columns={'History of Magic': 'History'}, inplace=True)
		return stats
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def main():
	if len(sys.argv) != 2:
		print(RED + "Pass Data to Describe!" + RESET)
		sys.exit(1)
	
	df = loadData(sys.argv[1])
	df = cleanData(df)
	stats = extractInfo(df)
	pd.options.display.float_format = '{:.6f}'.format
	print(stats)

if __name__ == "__main__":
	main()
