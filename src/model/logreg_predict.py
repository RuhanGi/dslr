from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import sys

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def getAccuracy():
    try:
        GREEN = "\033[92m"
        RESET = "\033[0m"
        truth_df = pd.read_csv("datasets/dataset_truth.csv")
        predictions_df = pd.read_csv("houses.csv")
        y_true = truth_df["Hogwarts House"].values
        y_pred = predictions_df["Hogwarts House"].values
        print(GREEN + f"Accuracy of {sys.argv[2]}: {accuracy_score(y_true, y_pred) * 100:.4f}%" + RESET)
    except Exception as e:
        print(RED + f"Error with loading accuracy files: {e}" + RESET)
        sys.exit(1)

def load(fil):
    try:
        return pd.read_csv(fil, index_col=0)
    except FileNotFoundError:
        print(RED + f"Error: {fil} not found!" + RESET)
        sys.exit(1)

def parseInputs(df, features):
    missing_cols = set(features) - set(df.columns)
    if missing_cols:
        print(RED + f"Error: Missing columns in dataset_test.csv: {missing_cols}" + RESET)
        sys.exit(1)

    # ? BASTARDE
    df.fillna(0, inplace=True)
    inputs = df[features].to_numpy()
    inputs = np.hstack((inputs, np.ones((inputs.shape[0], 1))))
    return inputs

def parseth(th):
    th.fillna(0)
    features = list(th.columns)
    features, label = features[:-1], features[-1]
    return features, label, list(th.index), th.to_numpy()

def softmax(weights, depen):
    z = np.dot(depen, weights.T)
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return np.argmax(exp / np.sum(exp, axis=1, keepdims=True), axis=1)

def main():
    if len(sys.argv) != 3:
        print(RED + "Usage: python estimate.py dataset_test.csv weights.csv" + RESET)
        sys.exit(1)

    df = load(sys.argv[1])
    th = load(sys.argv[2])
    features, label, unique, weights = parseth(th)
    inputs = parseInputs(df, features)

    pred = np.array(unique)[softmax(weights, inputs)]
    df = pd.DataFrame({label: pred}, index=df.index)
    df.to_csv("houses.csv")

    getAccuracy()

if __name__ == "__main__":
    main()
