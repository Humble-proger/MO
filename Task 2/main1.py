import numpy as np
from math import sqrt, acos, pi
import matplotlib.pyplot as plt
import pandas as pd

def f(vector):
    x, y = vector[0], vector[1]
    def Function1():
        return 100*(y-x)**2 + (1-x)**2
    def Function2():
        return 100*(y-x**2)**2 + (1-x)**2
    def Function3():
        return 1/(1 + pow((x-2)/3, 2) + pow((y - 2) / 3, 2)) + 3/(1 + pow(x-1, 2) + pow((y - 1)/2, 2))
    return Function3()

def f1(vector):
    x, y = vector[0], vector[1]
    def denominator1(x, y):
        return pow(1 + pow((x-2)/3, 2) + pow((y - 2) / 3, 2), 2)
    def denominator2(x, y):
        return pow(1 + pow(x-1, 2) + pow((y - 1)/2, 2), 2)
    def Function3():
        return np.array([-(2*x - 4)/(9 * denominator1(x, y)) - (6*(x - 1))/denominator2(x, y), 
                        -(2*y - 4)/(9 * denominator1(x, y)) - (3*(y - 1)) / (2*denominator2(x, y))])
    def Function1():
        return np.array([-200*y + 202*x - 2, 200*(y-x)])
    def Function2():
        return np.array([-400*(-x**2 + y)*x+2*x-2, 200*(y-x**2)])
    return Function3()

def find_alpha0(f, xk, pk, eps, min_alpha =1e-8, max_alpha = 50.0, old_fk = None, _direction=-1, maxiter=100):
    if old_fk == None:
        old_fk = f(xk)
    def func(alphak):
        return f(xk + alphak*pk)
    def golden_selection_maximization(a, b, eps):
        calc_func = 0
        val = (sqrt(5) - 1)/2
        x1 = a + (1 - val) * (b - a)
        x2 = a + val * (b - a)
        f1, f2 = func(x1), func(x2)
        calc_func += 2
        while abs(a - b) > eps:
            if f1 < f2:
                a = x1
                x1, x2 = x2, a + val * (b - a)
                f1, f2 = f2, func(x2)
                calc_func += 1
            else:
                b = x2
                x2, x1 = x1, a + (1 - val) * (b - a)
                f2, f1 = f1, func(x1)
                calc_func += 1
        return (a + b) / 2, calc_func
    def golden_selection_minimization(min_alpha, max_alpha, eps):
        a, b = min_alpha, max_alpha
        val = (sqrt(5) - 1)/2
        x1 = a + (1 - val) * (b - a)
        x2 = a + val * (b - a)
        f1, f2 = func(x1), func(x2)
        while abs(a - b) > eps:
            if f1 > f2:
                a = x1
                x1, x2 = x2, a + val * (b - a)
                f1, f2 = f2, func(x2)
            else:
                b = x2
                x2, x1 = x1, a + (1 - val) * (b - a)
                f2, f1 = f1, func(x1)
        return (a + b) / 2
    def parabolic_minimization(a, b, eps, max_iter=100):
        iter = 0
        calc_func = 0
        X = np.array([a, (a + b) / 2, b])
        F = np.array([func(X[0]), func(X[1]), func(X[2])])
        if max_iter < 1:
            return X[1], calc_func
        calc_func += 3
        while iter < max_iter:
            numerator = (X[1] - X[0])**2 * (F[1] - F[2]) - (X[1] - X[2])**2 * (F[1] - F[0])
            denominator = 2 * ((X[1] - X[0]) * (F[1] - F[2]) - (X[1] - X[2]) * (F[1] - F[0]))
            x_min = X[1] - numerator / denominator
            func_min = func(x_min)
            calc_func += 1
            _templist = sorted(zip([F[0], F[1], F[2], func_min], [X[0], X[1], X[2], x_min]), key=lambda pair: pair[0])
            if abs(min(F[0], F[1], F[2]) - func_min) / abs(func_min) < eps:
                return (_templist[0][1], calc_func)
            X = [elem[1] for elem in _templist[:3]]
            F = [elem[0] for elem in _templist[:3]]
            iter += 1
        return (_templist[0][1], calc_func)
    def parabolic_maximization(a, b, eps, max_iter=100):
        iter = 0
        calc_func = 0
        X = np.array([a, (a + b) / 2, b])
        F = np.array([func(X[0]), func(X[1]), func(X[2])])
        if max_iter < 1:
            return X[1], calc_func
        calc_func += 3
        while iter < max_iter:
            numerator = (X[1] - X[0])**2 * (F[1] - F[2]) - (X[1] - X[2])**2 * (F[1] - F[0])
            denominator = 2 * ((X[1] - X[0]) * (F[1] - F[2]) - (X[1] - X[2]) * (F[1] - F[0]))
            x_max = X[1] - numerator / denominator
            func_max = func(x_max)
            calc_func += 1
            _templist = sorted(zip([F[0], F[1], F[2], func_max], [X[0], X[1], X[2], x_max]), key=lambda pair: pair[0], reverse=True)
            if abs(max(F[0], F[1], F[2]) - func_max) / abs(func_max) < eps:
                return (_templist[0][1], calc_func)
            X = [elem[1] for elem in reversed(_templist[:3])]
            F = [elem[0] for elem in reversed(_templist[:3])]
            iter += 1
        return (_templist[0][1], calc_func)
    if _direction == -1:
        return parabolic_minimization(min_alpha, max_alpha, eps, max_iter=maxiter)
    else:
        alpha2, func_calc2 = golden_selection_maximization(min_alpha, max_alpha, eps)
        return alpha2, func_calc2

def norm(vector):
    calc = 0
    for i in vector:
        calc += i**2
    return sqrt(abs(calc))

def angle_XY_S(XY, S):
    XY = XY / norm(XY)
    S = S / norm(S)
    x, y, s1, s2 = XY[0], XY[1], S[0], S[1]
    def arg():
        return (x*s1+y*s2)
    a = arg()
    if a >= -1 and a <= 1: 
        return acos(arg())*180/pi
    if a > 1 and a <= 3:
        return acos(arg() - 2)*180/pi - 180
    if a < -1 and a >= -3:
        return acos(arg() + 2)*180/pi + 180

def angle_XY_S2(XY, S):
    XY = XY / norm(XY)
    S = S / norm(S)
    x, y, s1, s2 = XY[0], XY[1], S[0], S[1]
    def arg():
        return (x*s1+y*s2)
    return acos(arg())*180/pi

def matrix_to_string(vector, flag=True):
    if flag:
        out = ""
        for i in range(2):
            out += '[ '
            for j in range(2):
                out += "{0:.1f}".format(vector[i][j]) + " "
            out += "]\n"
    else:
        out = '[ '
        for i in range(2):
            out += "{0:.1f}".format(vector[i]) + " "
        out += "]\n"
    return out

def dfp_method(func, funcd, x0, maxiter=50, eps=1e-4, direction=-1, _maxalpha=50, _minalpha =1e-8, _table=False):
    if _table:
        df = pd.DataFrame(columns=np.array(["x", "y", 'f(x, y)', "s1", "s2", 'lambda','|xi - xm1|', '|yi - ym1|','|fi - fm1|', 'angle', 'gfk', 'H']))
    gfk = funcd(x0)
    N = len(x0)
    k = 0
    Hk = np.eye(N)
    xk = x0
    xkp1 = x0 + 2*eps
    alpha_k = 1.0
    calc_func = 0
    if _table:
        df.loc[len(df)] = np.array([x0[0], x0[1], func(x0), 0.0, 0.0, 1.0, abs(x0[0]), abs(x0[1]), abs(func(x0)), 0, matrix_to_string(gfk, False), matrix_to_string(Hk)])
    while norm(gfk) > eps and maxiter > k:
        pk = direction*np.dot(Hk, gfk) #Определяем направления спуска
        alpha_k, cfk = find_alpha0(func, xk, pk, 1e-7, _direction=direction, max_alpha=_maxalpha,min_alpha=_minalpha, maxiter = 100) #Нахождение альфа по условиям Вольфа. Если производная имеет больше одного нуля может не работать
        calc_func += cfk
        xkp1 = xk + alpha_k * pk
        gfkp1 = funcd(xkp1)
        sk = alpha_k * pk
        rk = gfkp1 - gfk

        sTs = np.outer(sk, sk)
        rTr = np.outer(rk, rk)

        #Считаем Гессиян k+1
        A = sTs / (sk @ rk)
        B = (Hk @ rTr @ Hk) / (rk @ Hk @ rk)
        Hk = Hk + A - B
        k += 1
        if _table:
            df.loc[len(df)] = np.array([xkp1[0], xkp1[1], func(xkp1), pk[0], pk[1], alpha_k, abs(xkp1[0]-xk[0]), abs(xkp1[1]-xk[1]), abs(func(xkp1)-func(xk)), angle_XY_S(xkp1-xk, pk), matrix_to_string(gfk, False), matrix_to_string(Hk)])
        xk = xkp1
        gfk = gfkp1
    if _table:
        return (xk, df)
    else:
        return xk, k, calc_func

def CG_method(func, funcd, x0, maxiter=50, eps=1e-4, direction=-1, _max_alpha = 100, _table=False):
    if _table:
        df = pd.DataFrame(columns=np.array(["x", "y", 'f(x, y)', "s1", "s2", 'lambda','|xi - xm1|', '|yi - ym1|','|fi - fm1|', 'angle', 'gfk']))
    k = 0
    calc_func = 0
    xk = x0
    alphak = 1.0
    gfk, gfkp1 = funcd(xk), 0
    pk = direction * gfk
    if _table:
        df.loc[len(df)] = np.array([x0[0], x0[1], func(x0), 0.0, 0.0, 1.0, abs(x0[0]), abs(x0[1]), abs(func(x0)), 0, "[0, 0]"])
    while norm(pk) > eps and k < maxiter:
        alphak, cfk = find_alpha0(func, xk, pk, 1e-3, _direction = direction, max_alpha=_max_alpha, maxiter=300)
        calc_func += cfk
        xkp1 = xk + alphak*pk
        gfkp1 = funcd(xkp1)
        w = np.dot(gfkp1, gfkp1) / np.dot(gfk, gfk)
        pk = direction*(gfkp1 - w * pk)
        if _table:
            df.loc[len(df)] = np.array([xk[0], xk[1], func(xk), pk[0], pk[1], alphak, abs(xkp1[0]-xk[0]), abs(xkp1[1]-xk[1]), abs(func(xkp1)-func(xk)), angle_XY_S(xkp1-xk, pk), matrix_to_string(pk, False)])
        xk = xkp1
        gfk = gfkp1
        k += 1
    if _table:
        return xk, df
    else:
        return xk, k, calc_func




def main():
    
    res1, df1 = dfp_method(f, f1, np.array([5, 3]), eps=1e-3, direction=1, maxiter=30, _maxalpha=100, _table=True)
    print(f(res1), res1.__repr__())
    #print(df)
    res2, df2 = CG_method(f, f1, np.array([5, 3]), eps=1e-3, direction=1, maxiter=30, _table=True, _max_alpha=200)
    print(f(res2), res2.__repr__())
    with pd.ExcelWriter("OutputFile.xlsx") as writer:
        df1.to_excel(writer, sheet_name="DFP Method", index_label='i', float_format="%.8f")
        df2.to_excel(writer, sheet_name="CG Method", index_label='i', float_format="%.8f")
    
    x = np.arange(-10, 10, 0.5)
    y = np.arange(-10, 10, 0.5)
    x_grid, y_grid = np.meshgrid(x, y)
    #z = 100*(y_grid-x_grid**2)**2 + (1-x_grid)**2
    #z = 100*(y_grid-x_grid)**2 + (1-x_grid)**2
    z = 1/(1 + ((x_grid-2)/3)**2 + ((y_grid - 2) / 3) ** 2) + 3/(1 + (x_grid-1)**2 + ((y_grid - 1)/2)**2)
    cs = plt.contour(x_grid, y_grid, z, zorder=1, levels=10)
    plt.clabel(cs)
    plt.plot(np.array(list(map(float, df1['x']))), np.array(list(map(float, df1['y']))), "->", zorder=2, color="red", label="DFP Method")
    plt.plot(np.array(list(map(float, df2['x']))), np.array(list(map(float, df2['y']))), "->", zorder=2, color="blue", label="CG Method")
    #plt.scatter(res1[0], res1[1])
    plt.title("Function level lines")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()