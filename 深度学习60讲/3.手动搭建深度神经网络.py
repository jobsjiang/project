# -*- coding：utf-8 -*-
import numpy as np
def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters
# def linear_activation_forward(A_prev,W,b,activation):
#     if activation == 'sigmoid':
        # Z,linear_cache = linear_forward(A_prev,W,b)