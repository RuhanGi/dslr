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

if __name__ == "__main__":
	main()
