# -*- coding：utf-8 -*-
import numpy as np
# 定义sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# 初始化参数
def initilize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0.0
    return w,b
# 定义前向传播以及损失函数
def propagate(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
    dw = np.dot(X,(A - Y).T) / m
    db = np.sum(A - Y) / m
    assert dw.shape == w.shape
    assert db.dtype == float
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {'dw':dw,'db':db}
    return grads,cost
# 定义反向传播
def backward_propagation(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs = []
    for i in range(num_iterations):
        grad,cost = propagate(w,b,X,Y)
        dw = grad['dw']
        db = grad['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i %100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("cost after iteration %i:%f"%(i,cost))
    params = {'dw':w,'db':b}
    grads = {'dw':dw,'db':db}
    return params,grads,costs
# 定义预测函数
def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X) + b)
    for i in range(A.shape[1]):
        if A[:,i] > 0.5:
            Y_prediction[:,i] = 1
        else:
            Y_prediction[:,i] = 0
    assert Y_prediction.shape == (1,m)
    return Y_prediction
# 封装函数
def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,learning_rate = 0.5,print_cost = False):
    # 初始化参数
    w,b = initilize_with_zeros(X_train.shape[0])
    parameters,grads,costs = backward_propagation(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w = parameters['w']
    b = parameters['b']
    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)
    print("train accuracy:{}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy:{}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d





