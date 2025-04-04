import pandas as pd
import numpy as np
import sys

from sklearn.metrics import accuracy_score, classification_report

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def load_thetas(filename):
	try:
		return pd.read_csv(filename, index_col=0)
	except FileNotFoundError:
		print(RED + f"Error: {filename} not found!" + RESET)
		sys.exit(1)

def load_test_data(filename):
	try:
		return pd.read_csv(filename, index_col=0)
	except FileNotFoundError:
		print(RED + f"Error: {filename} not found!" + RESET)
		sys.exit(1)

def softmax(z):
	exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True))
	return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def predict_classes(ndata, th):
	z = np.dot(ndata, th.T)
	probs = softmax(z)
	predictions = np.argmax(probs, axis=1)
	return predictions

def main():
	if len(sys.argv) != 3:
		print(RED + "Usage: python estimate.py dataset_test.csv thetas.csv" + RESET)
		sys.exit(1)

	th = load_thetas(sys.argv[2])

	mins = th.loc["mins"].values[1:]
	ranges = th.loc["ranges"].values[1:]
	th = th.drop(index=["mins", "ranges"])

	test_data = load_test_data(sys.argv[1])
	headers = th.columns[1:]
	missing_cols = set(headers) - set(test_data.columns)
	if missing_cols:
		print(RED + f"Error: Missing columns in dataset_test.csv: {missing_cols}" + RESET)
		sys.exit(1)

	data = test_data[headers].to_numpy()
	ndata = (data - mins) / ranges
	ndata = np.hstack((np.ones((ndata.shape[0], 1)), ndata))

	predicted_indices = predict_classes(ndata, th)
	house_labels = th.index[predicted_indices]

	predictions_df = pd.DataFrame({
		"Index": test_data.index,
		"Hogwarts House": house_labels
	})

	predictions_df.to_csv("houses.csv", index=False)
	if "Hogwarts House" in test_data.columns:
		actual_labels = test_data["Hogwarts House"].values
		accuracy = accuracy_score(actual_labels, house_labels)
		report = classification_report(actual_labels, house_labels, target_names=th.index)

		print(GREEN + f"\nModel Evaluation:\nAccuracy: {accuracy:.4f}\n" + RESET)

if __name__ == "__main__":
	main()
