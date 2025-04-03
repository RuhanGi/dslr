import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import csv

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
PURPLE = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[97m"
BLACK = "\033[98m"
RESET = "\033[0m"

def load():
	try:
		return pd.read_csv("thetas.csv", index_col=0)
	except:
		print(RED + "No properly trained file found!" + RESET)

def get(name):
	try:
		return float(input(BLUE + name + ": " + YELLOW))
	except:
		print(RED + "Input a Number!" + RESET)
		return get(name)

def softmax(z):
	exp_scores = np.exp(z - np.max(z, axis=1, keepdims=True)) 
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	return probs

def predict_class(ndata, th):
	z = np.dot(ndata, th.T)
	probs = softmax(z)
	predicted_class_index = np.argmax(probs)
	return predicted_class_index, probs[0][predicted_class_index]

def main():
	th = load()
	headers = th.columns[1:]
	
	# Prepare the input vector for prediction
	n = len(headers)  # Number of features
	ndata = np.ones((1, n + 1))  # Adding bias term (1 for bias)
	
	# Get input for each feature
	for i, header in enumerate(headers):
		ndata[0, i + 1] = get(header)
	
	# Predict the class
	predicted_class_index, probability = predict_class(ndata, th)
	
	# Display the result
	print(GREEN + f"Predicted Class: {th.index[predicted_class_index]} with Probability: {probability:.4f}" + RESET)


if __name__ == "__main__":
	main()