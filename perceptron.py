import numpy as np

def perceptron(X, y, k):
    T = len(X)
    d = len(X[0])
    W = np.zeros((k, d))
    error_count = np.zeros(T)
    for t in range(len(X)):
        y_hat = np.argmax(np.matmul(W, X[t]))
        if y_hat != y[t]:
            W[y[t]] = W[y[t]] + X[t]
            W[y_hat] = W[y_hat] - X[t]
            if t == 0:
                error_count[t] = 1
            else:
                error_count[t] = error_count[t - 1] + 1
        else:
            if t == 0:
                error_count[t] = 1
            else:
                error_count[t] = error_count[t - 1]
    return error_count