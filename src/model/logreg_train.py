import matplotlib.pyplot as plt
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

        # features = list(df.select_dtypes(include=['float64']).columns)
        features = ['Defense Against the Dark Arts', 'Charms', 'Flying']
        # features = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 
        #                'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 
        #                'Transfiguration', 'Charms', 'Flying']
        label = 'Hogwarts House'

        df.dropna(inplace=True, subset=features+[label])
        df.drop_duplicates(inplace = True)
    
        return features, np.array(df[features]), label, np.array(df[label])
    except Exception as e:
        print(RED + "Error: " + str(e) + RESET)
        sys.exit(1)

def categoricalCrossentropy(onehot, probs):
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    return -np.mean(np.sum(onehot * np.log(probs), axis=1))

def softmax(weights, depen):
    z = np.dot(depen, weights.T)
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def epoch(weights, depen, onehot, learningRate):
    probs = softmax(weights, depen)
    grad = np.dot((probs - onehot).T, depen) / depen.shape[0]
    weights -= learningRate * grad
    return categoricalCrossentropy(onehot, probs)

def trainModel(features, depen, indep):
    # try:
    means = np.average(depen, axis=0)
    stds = np.std(depen, axis=0)
    stds[stds == 0] = 1
    depen = (depen - means) / stds

    unique = np.unique(indep)
    mapper = {house: i for i, house in enumerate(unique)}
    indep = np.vectorize(mapper.get)(indep)
    onehot = np.eye(len(unique))[indep]

    depen = np.hstack((depen, np.ones((depen.shape[0], 1))))
    weights = np.zeros((unique.shape[0], depen.shape[1]))

    learningRate = 0.3
    maxepochs = 100000
    tolerance = 10**-5

    # for e in range(maxepochs):
    #     prvweights = weights.copy()
    #     loss = epoch(weights, depen, onehot, learningRate)
    #     maxDiff = np.max(np.abs(weights-prvweights))
    #     print(f"\rEpoch [{e}/{maxepochs}]: Loss={loss:.6f}, Diff={maxDiff:.6f}", end="")
    #     if maxDiff < tolerance:
    #         print()
    #         break
    for e in range(maxepochs):
        prvweights = weights.copy()
        # loss = epoch(weights, depen, onehot, learningRate)
        batch_size = 8
        for i in range(0, len(depen), batch_size):
            epoch(weights, depen[i:i+batch_size], onehot[i:i+batch_size], learningRate)
        maxDiff = np.max(np.abs(weights-prvweights))
        # print(f"\rEpoch [{e}/{maxepochs}]: Loss={loss:.6f}, Diff={maxDiff:.6f}", end="")
        print(f"\rEpoch [{e}/{maxepochs}]: Diff={maxDiff:.6f}", end="")
        if maxDiff < tolerance:
            print()
            break
    print(GREEN + "\rModel Trained!" + (" " * 50) + RESET)
    
    weights[:, :-1], weights[:,-1] = weights[:, :-1] / stds, \
        weights[:, -1] - np.sum(weights[:, :-1] * means / stds, axis=1)

    return unique, weights
    # except Exception as e:
    #     print(RED + "Error: " + str(e) + RESET)
    #     sys.exit(1)

def main():
    if len(sys.argv) != 2:
        print(RED + "Pass Training Data!" + RESET)
        sys.exit(1)
    
    features, depen, label, indep = loadData(sys.argv[1])
    unique, weights = trainModel(features, depen, indep)

    df = pd.DataFrame(weights, index=unique, columns=features+[label])
    df.to_csv("weights.csv")

if __name__ == "__main__":
    main()