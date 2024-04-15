import numpy as np
from math import exp

def f(x):
    return (x - 5)**2 + 10
def parabolic_minimization(func, a, b, eps, max_iter=100):
    iter = 0
    X = np.array([a, (a + b) / 2, b])
    F = np.array([func(X[0]), func(X[1]), func(X[2])])
    if max_iter < 1:
        return X[1]
    while iter < max_iter:
        numerator = (X[1] - X[0])**2 * (F[1] - F[2]) - (X[1] - X[2])**2 * (F[1] - F[0])
        denominator = 2 * ((X[1] - X[0]) * (F[1] - F[2]) - (X[1] - X[2]) * (F[1] - F[0]))
        x_min = X[1] - numerator / denominator
        func_min = func(x_min)
        _templist = sorted(zip([F[0], F[1], F[2], func_min], [X[0], X[1], X[2], x_min]), key=lambda pair: pair[0])
        if abs(min(F[0], F[1], F[2]) - func_min) / abs(func_min) < eps:
            return _templist[0][1]
        X = [elem[1] for elem in _templist[:3]]
        F = [elem[0] for elem in _templist[:3]]
        iter += 1
    return _templist[0][1]
def parabolic_maximization(func, a, b, eps, max_iter=100):
    iter = 0
    X = np.array([a, (a + b) / 2, b])
    F = np.array([func(X[0]), func(X[1]), func(X[2])])
    if max_iter < 1:
        return X[1]
    while iter < max_iter:
        numerator = (X[1] - X[0])**2 * (F[1] - F[2]) - (X[1] - X[2])**2 * (F[1] - F[0])
        denominator = 2 * ((X[1] - X[0]) * (F[1] - F[2]) - (X[1] - X[2]) * (F[1] - F[0]))
        x_max = X[1] - numerator / denominator
        func_max = func(x_max)
        max_ind, max_val = max(enumerate(F), key=lambda pair: pair[1])
        if x_max > b or x_max < a:
            return X[max_ind]
        if abs(max_val - func_max) / abs(func_max) < eps:
            return x_max if func_max > max_val else X[max_ind]
        if max_val > func_max:
            if X[max_ind] < x_max:
                X = np.array([X[max_ind-1], X[max_ind], x_max])
                F = np.array([F[max_ind-1], F[max_ind], func_max])
            else:
                X = np.array([x_max, X[max_ind], X[max_ind+1]])
                F = np.array([func_max, F[max_ind], F[max_ind+1]])
        else:
            if X[max_ind] < x_max:
                X = np.array([X[max_ind], x_max, X[max_ind+1]])
                F = np.array([F[max_ind], func_max, F[max_ind+1]])
            else:
                X = np.array([X[max_ind-1], x_max, X[max_ind]])
                F = np.array([F[max_ind-1], func_max, F[max_ind]])
        iter += 1
    return x_max if func_max > max_val else X[max_ind]

x = parabolic_minimization(f, -2, 8, 1e-4)
print(x, f(x))