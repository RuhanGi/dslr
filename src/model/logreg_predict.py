from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import sys

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def loadData(fil):
	try:
		return pd.read_csv(fil, index_col=0)
	except FileNotFoundError:
		print(RED + f"Error: {fil} not found!" + RESET)
		sys.exit(1)

def loadThetas(fil):
	try:
		return pd.read_csv(fil, index_col=0)
	except FileNotFoundError:
		print(RED + f"Error: {fil} not found!" + RESET)
		sys.exit(1)

def softmax(z):
	exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
	return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def predict_classes(ndata, th):
	z = np.dot(ndata, th.T)
	probs = softmax(z)
	return np.argmax(probs, axis=1)

import seaborn as sns
import matplotlib.pyplot as plt

def plotPairPlot(predictions_df, test_data, th):
	try:
		# Load true labels
		truth_df = pd.read_csv("datasets/dataset_truth.csv")

		# Merge predictions with true values for comparison
		merged = predictions_df.copy()
		merged["True House"] = truth_df["Hogwarts House"].values

		# Determine correctness
		merged["Status"] = np.where(
			merged["Hogwarts House"] != merged["True House"],
			"Incorrect",
			merged["Hogwarts House"]
		)

		# Extract feature columns from thetas
		feature_cols = th.columns[1:]  # skip bias
		course_data = test_data[feature_cols].copy()
		course_data["Status"] = merged["Status"]
		course_data["True House"] = merged["True House"]

		# Create pairplot
		pair = sns.pairplot(course_data, hue="Status", palette="husl", diag_kind="kde", plot_kws={"alpha": 0.7})
		plt.suptitle("Pair Plot with Incorrect Predictions Labeled", y=1.02)

		# Annotate incorrect points with true labels
		incorrect_data = course_data[course_data["Status"] == "Incorrect"]

		for i, row_i in enumerate(pair.x_vars):
			for j, row_j in enumerate(pair.y_vars):
				if i < j:
					ax = pair.axes[j, i]
					for idx, row in incorrect_data.iterrows():
						ax.text(
							row[row_i],
							row[row_j],
							row["True House"],
							color="black",
							fontsize=8,
							ha="center",
							va="center"
						)

		plt.tight_layout()
		plt.show()

	except Exception as e:
		print(RED + f"Error generating pair plot: {e}" + RESET)
		sys.exit(1)

# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# def getAccuracy(predictions_df):
# 	try:
# 		truth_df = pd.read_csv("datasets/dataset_truth.csv")
# 		y_true = truth_df["Hogwarts House"].values
# 		y_pred = predictions_df["Hogwarts House"].values
# 		acc = accuracy_score(y_true, y_pred) * 100
# 		labels = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
# 		cm = confusion_matrix(y_true, y_pred, labels=labels)
# 		display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
# 		display.plot(cmap=plt.cm.Blues, xticks_rotation=45)
# 		plt.title(f'Confusion Matrix ({acc:.2f}% Accuracy)')
# 		plt.tight_layout()
# 		plt.show()
# 		return acc
# 	except Exception as e:
# 		print(RED + f"Error loading truth file: {e}" + RESET)
# 		sys.exit(1)

def getAccuracy(predictions_df):
	try:
		truth_df = pd.read_csv("datasets/dataset_truth.csv")
		y_true = truth_df["Hogwarts House"].values
		y_pred = predictions_df["Hogwarts House"].values
		return accuracy_score(y_true, y_pred) * 100
	except Exception as e:
		print(RED + f"Error loading truth file: {e}" + RESET)
		sys.exit(1)
	
def main():
	if len(sys.argv) != 3:
		print(RED + "Usage: python estimate.py dataset_test.csv thetas.csv" + RESET)
		sys.exit(1)

	test_data = loadData(sys.argv[1])
	th = loadThetas(sys.argv[2])

	headers = th.columns[1:]
	missing_cols = set(headers) - set(test_data.columns)
	if missing_cols:
		print(RED + f"Error: Missing columns in dataset_test.csv: {missing_cols}" + RESET)
		sys.exit(1)

	data = test_data[headers].to_numpy()
	data = np.hstack((np.ones((data.shape[0], 1)), data))
	predicted_indices = predict_classes(data, th)

	house_labels = th.index[predicted_indices]

	predictions_df = pd.DataFrame({
		"Index": test_data.index,
		"Hogwarts House": house_labels
	})

	predictions_df.to_csv("houses.csv", index=False)
	print(GREEN + f"Model Evaluation:\nAccuracy: {getAccuracy(predictions_df):.4f}%" + RESET)

	plotPairPlot(predictions_df, test_data, th)

if __name__ == "__main__":
	main()
