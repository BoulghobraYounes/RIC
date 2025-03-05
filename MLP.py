import numpy as np

def initialization(network: tuple) -> dict:
    parameters = {}
    L = len(network)
    for layer in range(1, L):
        parameters[f"W{layer}"] = np.random.randn(network[layer], network[layer - 1])
        parameters[f"b{layer}"] = np.zeros((network[layer], 1))
    return parameters

network = (2, 4, 4, 1)
parameters = initialization(network)
for key, value in parameters.items():
    print(key, value.shape)