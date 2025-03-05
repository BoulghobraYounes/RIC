import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def initialization(network: tuple) -> dict[str, np.ndarray]:
    parameters = {}
    L = len(network)
    for layer in range(1, L):
        parameters[f"W{layer}"] = np.random.randn(network[layer], network[layer - 1])
        parameters[f"b{layer}"] = np.zeros((network[layer], 1))
    return parameters


# Activation function
def sigmoid(potential: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-potential))


def forward_propagation(parameters: dict[str, np.ndarray], X: np.ndarray) -> dict[str, np.ndarray]:
    activations = {'A0': X}
    L = len(parameters) // 2
    for layer in range(1, L + 1):
        potential = np.dot(parameters[f"W{layer}"], activations[f"A{layer - 1}"]) + parameters[f"b{layer}"]
        activations[f"A{layer}"] = sigmoid(potential)
    return activations


# Loss function
def mean_squared_error(A, y):
    m = y.shape[1]
    return np.sum(0.5 * (A - y)**2) / m 


def back_propagation_v0(X, y, activations, parameters):
    A2 = activations['A2'] # (1, m)
    A1 = activations['A1'] # (2, m)
    W2 = parameters['W2'] # (1, 2)

    m = y.shape[1] # number of samples (for normalization)
    
    # (1, m)
    Delta2 = 1 / m * (A2 - y) * A2 * (1 - A2) # A2 - y => (1, m) / A2*(1 - A2) => (1, m) ==> Product element by element
    # (2, m)
    Delta1 = np.dot(W2.T, Delta2) * A1 * (1 - A1) # W2.T * delta2 => (2, m) / A1 * (1 - A1) => (2, m) ==> Product element by element
    
    dW2 = np.dot(Delta2, A1.T) # (1, 2)
    db2 = np.sum(Delta2, axis=1, keepdims=True) # (1, 1)
    dW1 = np.dot(Delta1, X.T) # X => (2, m) ===> (2, 2)
    db1 = np.sum(Delta1, axis=1, keepdims=True) # (2, 1)

    gradients = {
        'dW2': dW2,
        'db2': db2,
        'dW1': dW1,
        'db1': db1
    }
    return gradients


def update_weights(parameters, gradients, learning_rate):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parameters


def two_layers_neural_network(X, y, learning_rate = 0.01, n_iterations = 1000):
    network = (2, 2, 1)
    parameters = initialization(network)
    loss = []
    
    for _ in tqdm(range(n_iterations)):
        # Learning
        activations = forward_propagation(parameters, X)
        A2 = activations['A2']
        gradients = back_propagation_v0(X, y, activations, parameters)
        parameters = update_weights(parameters, gradients, learning_rate)

        # Store Loss 
        loss.append(mean_squared_error(A2, y))

    plt.plot(loss, label='train loss')
    plt.show()

    return parameters


X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]]) 

y = np.array([[0, 1, 1, 0]])

parameters = two_layers_neural_network(X, y, learning_rate=0.1, n_iterations=10000)
activations = forward_propagation(parameters, X)
print(activations['A2'])