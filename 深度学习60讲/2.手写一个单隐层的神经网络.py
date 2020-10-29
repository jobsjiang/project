# -*- coding：utf-8 -*-
import numpy as np


# 定义网络结构的函数
def layer_sizes(X,Y):
    # size of input layer
    n_x = X.shape[0]
    # size of hidden layer
    n_h = 4
    # size of output layer
    n_y = Y.shape[0]
    return n_x,n_h,n_y
# 初始化模型参数
def initialize_parameters(n_x,n_h,n_y):
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    parameters = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2
    }
    return parameters
