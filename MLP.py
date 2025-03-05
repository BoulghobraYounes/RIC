import numpy as np
from sklearn.datasets import make_blobs 


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



network = (2, 4, 4, 1)
parameters = initialization(network)

# Generate random dataset : X(100, 2) y(100)
X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
X = X.T # X => (2, 100)
activations = forward_propagation(parameters, X)
for key, value in activations.items():
    print(key, value.shape)
