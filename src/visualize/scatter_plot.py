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
		float_cols = df.select_dtypes(include=['float64'])
		df.dropna(inplace=True, subset=float_cols.columns)
		df.drop_duplicates(inplace = True)
		return df
	except Exception as e:
		print(RED + "Error: " + str(e) + RESET)
		sys.exit(1)

def plotData(df):
	numeric_df = df.select_dtypes(include=['float64'])
	corr_matrix = numeric_df.corr()
	corr_unstacked = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()
	top_correlations = corr_unstacked.abs().nlargest(4).index
 
	fig = plt.figure(figsize=(22, 11))
	gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1])

	ax0 = fig.add_subplot(gs[:, 0])
	mask = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)
	sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f",
				linewidths=0.5, annot_kws={"size": 10}, ax=ax0)
	ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45, ha='right', fontsize=10)
	ax0.set_yticklabels(ax0.get_yticklabels(), fontsize=10)
	ax0.set_title("Correlation Matrix", fontsize=14)

	for i, (var1, var2) in enumerate(top_correlations):
		ax = fig.add_subplot(gs[i // 2, 1 + i % 2])
		sns.scatterplot(x=df[var1], y=df[var2], ax=ax)
		correlation_value = corr_matrix.loc[var1, var2]
		ax.set_title(f"{var1} vs {var2}\n r = {correlation_value:.2f}")
		ax.set_xlabel(var1)
		ax.set_ylabel(var2)

	plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)
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