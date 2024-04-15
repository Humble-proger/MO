import numpy as np
from math import sqrt
from scipy.optimize import line_search, minimize

def f(vector):
    x, y = vector[0], vector[1]
    #return x**2 - x*y + y ** 2 + 9*x - 6*y + 20
    return 1/(1 + pow((x-2)/3, 2) + pow((y - 2) / 3, 2)) + 3/(1 + pow(x-1, 2) + pow((y - 1)/2, 2))

def f1(vector):
    x, y = vector[0], vector[1]
    #return np.array([2*x-y+9, -x+2*y-6])
    def denominator1(x, y):
        return pow(1 + pow((x-2)/3, 2) + pow((y - 2) / 3, 2), 2)
    def denominator2(x, y):
        return pow(1 + pow(x-1, 2) + pow((y - 1)/2, 2), 2)
    return np.array([-(2*x - 4)/(9 * denominator1(x, y)) - (6*(x - 1))/denominator2(x, y), 
                    -(2*y - 4)/(9 * denominator1(x, y)) - (3*(y - 1)) / (2*denominator2(x, y))])

def find_direction(func, x, delta):
    if func(x) > func(x + delta):
        return delta
    return -delta

def find_min_interval(func, f,fg, yk, pk, alpha, delta):
    if func(f,fg, yk, pk, alpha) <= func(f,fg, yk, pk, alpha + delta):
        delta *= -1
    
    h = 2 * delta
    alphak = alpha + h
    
    while func(f,fg, yk, pk, alpha) > func(f,fg, yk, pk, alphak):
        alpha, alphak = alphak, alphak + h
    #print(f'Минимальный отрезок: [{x - h}, {xk}], кол-во итераций: {k}')
    if alpha > alphak:
        return (alpha - h, alphak)
    else:
        return (alphak, alpha - h)
def find_min(func, f,fg, yk, pk, alpha, delta):
    if func(f,fg, yk, pk, alpha) <= func(f,fg, yk, pk, alpha + delta):
        delta *= -1
    alphak = alpha + delta
    
    while func(f,fg, yk, pk, alpha) > 0:
        alpha, alphak = alphak, alphak + delta
    #print(f'Минимальный отрезок: [{x - h}, {xk}], кол-во итераций: {k}')
    return alpha
def golden_selection(func, f, fg, xk, pk, alpha, eps):
    a, b = find_min_interval(func, f, fg, xk, pk, alpha, 0.5)
    val = (sqrt(5) - 1)/2
    x1 = a + (1 - val) * (b - a)
    x2 = a + val * (b - a)
    f1, f2 = func(f,fg, xk, pk, x1), func(f,fg, xk, pk, x2)
    while abs(a - b) > eps:
        a_step, b_step = a, b
        if f1 > f2:
            a = x1
            x1, x2 = x2, a + val * (b - a)
            f1, f2 = f2, func(f,fg, xk, pk, x2)
        else:
            b = x2
            x2, x1 = x1, a + (1 - val) * (b - a)
            f2, f1 = f1, func(f,fg, xk, pk, x1)
    
    print((a + b) / 2)
    return (a + b) / 2


def find_alpha0(f, fg, xk, pk, eps, alpha0 =1.0, c1 = 0.0001, c2=0.9):
    func = lambda f,fg, xk, pk, alpha: f(xk + alpha * pk) - f(xk) - c1*alpha*np.dot(pk, fg(xk))
    alpha = golden_selection(func, f, fg, xk, pk, alpha0, eps)
    if func(f,fg, xk, pk, alpha) > 0:
        raise "Не удачный начальный alpha"
    func = lambda f, fg, xk, pk, alpha: c2*np.dot(pk, fg(xk)) - np.dot(pk, fg(xk + alpha*pk))
    if func(f,fg, xk, pk, alpha) > 0:
        alpha = find_min(func, f, fg, xk, pk, alpha, eps)
    else:
        return alpha
    if func(f,fg, xk, pk, alpha) > 0:
        raise "Не удачный начальный alpha"
    return alpha

def norm(vector):
    calc = 0
    for i in vector:
        calc += i**2
    return sqrt(calc)

def outer_product(vector1, vector2):
    return vector1[:, np.newaxis] * vector2[np.newaxis, :]

def bfgs_method(func, funcd, x0, maxiter=1000, eps=1e-4):
    gfk = funcd(x0)
    N = len(x0)
    k = 0
    I = np.eye(N)
    Hk = I
    xk = x0
    alpha_k = 1.0
    while norm(gfk) > eps and k < maxiter and alpha_k > 0:
        pk = -np.dot(Hk, gfk)
        alpha_k = line_search(func, funcd, xk, pk)[0]
        xkp1 = xk + alpha_k * pk
        sk = alpha_k * pk
        xk = xkp1
        gfkp1 = funcd(xkp1)
        yk = gfkp1 - gfk
        gfk = gfkp1
        
        k += 1
        #Считаем Гессиян k+1
        r = 1.0 / np.dot(yk, sk)
        A = I - r * outer_product(sk, yk)
        B = I - r * outer_product(yk, sk)
        Hk = np.dot(A, np.dot(Hk, B)) + r * outer_product(sk, sk)
    return xk

def main():
    res = bfgs_method(f, f1, np.array([1, 1]), eps=1e-7)
    res1 = minimize(f, np.array([1, 1]), method="BFGS", jac=1e-7)
    print(res, res1)


if __name__ == "__main__":
    main()