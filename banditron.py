import numpy as np
import random

def banditron(X, y, k, **kwargs):
    T = len(X)
    d = len(X[0])
    W = np.zeros((k, d))
    error_count = np.zeros(T)

    if "gammas" not in kwargs:
        gammas = [kwargs["gamma"] for i in range(T)]
    else:
        gammas = kwargs["gammas"]
    
    if "eta" in kwargs:
        eta = kwargs["eta"]

    for t in range(len(X)):
        gamma = gammas[t]
        y_hat = np.argmax(np.matmul(W, X[t]))
        p = [gamma/k for i in range(k)]
        p[y_hat] = p[y_hat] + 1 - gamma
        y_tilde = random.choices(range(k), weights=p, k=1)[0]
        if y_tilde != y[t]:
            W[y_hat] = W[y_hat] - X[t]
            if t == 0:
                error_count[t] = 1
            else:
                error_count[t] = error_count[t - 1] + 1
        else:
            W[y_hat] = W[y_hat] - X[t]
            W[y_tilde] = W[y_tilde] + X[t] / p[y_tilde]
            if t == 0:
                error_count[t] = 1
            else:
                error_count[t] = error_count[t - 1]
    return error_count