import pandas as pd
import numpy as np
import time
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

def adagradEpoch(weights, inputs, onehot, learningRate, cache):
    probs = softmax(weights, inputs)
    grad = np.dot((probs - onehot).T, inputs) / inputs.shape[0]
    cache += grad**2
    epsilon = 10**-8
    learningRate = learningRate / (np.sqrt(cache) + epsilon)
    weights -= learningRate * grad
    return categoricalCrossentropy(onehot, probs), cache

def rmspropEpoch(weights, inputs, onehot, learningRate, cache):
    probs = softmax(weights, inputs)
    grad = np.dot((probs - onehot).T, inputs) / inputs.shape[0]

    decay, epsilon = 0.95, 10**-8
    cache = cache * decay + (1-decay) * grad**2
    learningRate = 0.05 / (np.sqrt(cache) + epsilon)
    weights -= learningRate * grad
    return categoricalCrossentropy(onehot, probs), cache

def adamEpoch(weights, inputs, onehot, learningRate, velocity, momentum, t):
    probs = softmax(weights, inputs)
    grad = np.dot((probs - onehot).T, inputs) / inputs.shape[0]
    decay1, decay2, epsilon = 0.9, 0.99, 10**-8
    momentum = decay1 * momentum + (1-decay1) * grad
    velocity = decay2 * velocity + (1-decay2) * grad**2
    learningRate = learningRate / (np.sqrt(velocity / (1-decay2**t)) + epsilon)
    weights -= learningRate * grad
    return categoricalCrossentropy(onehot, probs), velocity, momentum

def trainModel(inputs, onehot, optimizer):
    try:
        means = np.average(inputs, axis=0)
        stds = np.std(inputs, axis=0)
        stds[stds == 0] = 1
        inputs = (inputs - means) / stds    
        inputs = np.hstack((inputs, np.ones((inputs.shape[0], 1))))
        weights = np.zeros((onehot.shape[1], inputs.shape[1]))

        learningRate = 0.05
        maxepochs = 10000
        tolerance = 10**-3

        cache = np.zeros_like(weights)
        momentum = np.zeros_like(weights)
        velocity = np.zeros_like(weights)
        e = 0
        start_time = time.time()
        while e < maxepochs:
            prv = weights.copy()
            loss = epoch(weights, inputs, onehot, learningRate)

            if optimizer == 'batch':
                loss = epoch(weights, inputs, onehot, learningRate)
            elif optimizer == 'stochastic':
                for i in range(0, len(inputs)):
                    loss = epoch(weights, inputs[i:i+1], onehot[i:i+1], learningRate)
            elif optimizer == 'minibatch':
                batch_size = 32
                for i in range(0, len(inputs), batch_size):
                    loss = epoch(weights, inputs[i:i+batch_size], onehot[i:i+batch_size], learningRate)
            elif optimizer == 'adagrad':
                loss, cache = adagradEpoch(weights, inputs, onehot, learningRate, cache)
            elif optimizer == 'rmsprop':
                loss, cache = rmspropEpoch(weights, inputs, onehot, learningRate, cache)
            elif optimizer == 'adam':
                loss, velocity, momentum = adamEpoch(weights, inputs, onehot, learningRate, velocity, momentum, e+1)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
            e += 1
            maxDiff = np.max(np.abs(weights-prv))
            print(f"\rEpoch [{e}/{maxepochs}]: Loss={loss:.6f}, Diff={maxDiff:.6f}", end="")
            if maxDiff < tolerance:
                print()
                break

        duration = time.time() - start_time
        if e < maxepochs:
            print(GREEN + f"\r[{optimizer.upper()}] Model Trained in {duration:.4f} seconds!"+ RESET)
        else:
            print(YELLOW + f"\r[{optimizer.upper()}] Model Trained in {duration:.4f} seconds!" + " "*30 + RESET)
        
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

    weights = trainModel(inputs, onehot, 'batch')
    pd.DataFrame(weights, index=unique, columns=features+[label]).to_csv("batch.csv")

    weights = trainModel(inputs, onehot, 'stochastic')
    pd.DataFrame(weights, index=unique, columns=features+[label]).to_csv("stochastic.csv")

    weights = trainModel(inputs, onehot, 'minibatch')
    pd.DataFrame(weights, index=unique, columns=features+[label]).to_csv("minibatch.csv")

    weights = trainModel(inputs, onehot, 'adagrad')
    pd.DataFrame(weights, index=unique, columns=features+[label]).to_csv("adagrad.csv")

    weights = trainModel(inputs, onehot, 'rmsprop')
    pd.DataFrame(weights, index=unique, columns=features+[label]).to_csv("rmsprop.csv")

    weights = trainModel(inputs, onehot, 'adam')
    pd.DataFrame(weights, index=unique, columns=features+[label]).to_csv("adam.csv")

if __name__ == "__main__":
    main()