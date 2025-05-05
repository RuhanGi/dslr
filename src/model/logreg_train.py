import pandas as pd
import numpy as np
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
        df = pd.read_csv(fil)
        df = df.fillna(0)
        df.drop_duplicates(inplace = True)

        label = 'Hogwarts House'
        features = list(df.select_dtypes(include=['float64']).columns)
        indistincts =  ['Arithmancy', 'Potions', 'Care of Magical Creatures', 'Astronomy']
        for i in indistincts:
            features.remove(i)

        indep = np.array(df[label])
        unique = np.unique(indep)
        mapper = {house: i for i, house in enumerate(unique)}
        indep = np.vectorize(mapper.get)(indep)
        indep = np.eye(len(unique))[indep]

        return features, np.array(df[features]), label, indep, unique
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def categoricalCrossentropy(onehot, probs):
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    return -np.mean(np.sum(onehot * np.log(probs), axis=1))

def softmax(weights, inputs):
    z = np.dot(inputs, weights.T)
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def epoch(weights, inputs, onehot, learningRate):
    probs = softmax(weights, inputs)
    grad = np.dot((probs - onehot).T, inputs) / inputs.shape[0]
    weights -= learningRate * grad
    return categoricalCrossentropy(onehot, probs)

def trainModel(inputs, onehot):
    try:
        means = np.average(inputs, axis=0)
        stds = np.std(inputs, axis=0)
        stds[stds == 0] = 1
        inputs = (inputs - means) / stds    
        inputs = np.hstack((inputs, np.ones((inputs.shape[0], 1))))
        weights = np.zeros((onehot.shape[1], inputs.shape[1]))

        learningRate = 0.3
        maxepochs = 10000
        tolerance = 10**-4
        for e in range(maxepochs):
            prv = weights.copy()
            loss = epoch(weights, inputs, onehot, learningRate)
            maxDiff = np.max(np.abs(weights-prv))
            print(f"\rEpoch [{e}/{maxepochs}]: Loss={loss:.6f}, Diff={maxDiff:.6f}", end="")
            if maxDiff < tolerance:
                print()
                break
        print(GREEN + "\rModel Trained!" + (" " * 50) + RESET)

        weights[:, :-1], weights[:,-1] = weights[:, :-1] / stds, \
            weights[:, -1] - np.sum(weights[:, :-1] * means / stds, axis=1)

        return weights
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print(RED + "Pass Training Data!" + RESET)
        sys.exit(1)
    
    features, inputs, label, onehot, unique = loadData(sys.argv[1])  
    weights = trainModel(inputs, onehot)
    pd.DataFrame(weights, index=unique, columns=features+[label]).to_csv("weights.csv")

if __name__ == "__main__":
    main()