import pandas as pd
import numpy as np
from math import log, ceil, sin, cos, radians, sqrt

Name_OutFile = "OutputFile"
var_c = np.array([2, 4, 2, 6, 2, 3], dtype=np.int8)
var_a = np.array([-3, -6, 2, 6, -3, 8], dtype=np.int8)
var_b = np.array([6, -8, -8, 8, -4, -1], dtype=np.int8)


def func(x):
    if len(x) != 2:
        return 0
    subfunc = lambda x, y, a, b, c: c/(1 + (x - a)**2 + (y - b)**2)
    res = 0.0
    for i in range(6):
        res += subfunc(x[0], x[1], var_a[i], var_b[i], var_c[i])
    return -res

def find_N(P, Peps):
        if P > 1 or Peps > 1:
            return 0
        return ceil(log(1-P)/log(1-Peps))

def out_exel(df, name : str):
    if Name_OutFile != "":
        with pd.ExcelWriter(Name_OutFile + ".xlsx") as writer:
            df.to_excel(writer, sheet_name=name, float_format="%.8f", index=False)

def simple_random_search(p : float, pe : float, xmin = None):
    N = find_N(p, pe)
    if N == 0:
        return 0
    if xmin is None:
        xmin = np.random.uniform(-10, 10, 2)
    ymin = func(xmin)
    for i in range(N):
        xi = np.random.uniform(-10, 10, 2)
        yi = func(xi)
        if (yi < ymin):
            xmin, ymin = xi, yi
    return xmin, N + 1

def static_grad_medhod(xk, m : int, h : float = 0.5, eps : float = 0.1, maxiter : int = 1000):
    if m < 2:
        return 0
    
    gk = np.array([[0.0, 1.0]])
    for i in range(1, m):
        psi = radians(360*i/m)
        gk = np.concatenate((gk, np.array([[sin(psi), cos(psi)]])), axis=0)
    
    def calc_dgrad(xi):
        dgrad=np.array([0.0, 0.0])
        fxi = func(xi)
        for i in range(m):
            dgrad += gk[i] * (func(xk + eps*gk[i]) - fxi)
        return dgrad
    def norm(vector):
        calc = 0
        for i in vector:
            calc += i**2
        return sqrt(calc)
    
    dgrad, k = calc_dgrad(xk), 0
    norm_dgrad = norm(dgrad)
    count_calc_func = m + 1
    while norm_dgrad > eps and k < maxiter:
        xkp1 = xk - h*dgrad/norm_dgrad
        if xkp1[0] > 10:
            xkp1[0] = 10
        if xkp1[0] < -10:
            xkp1[0] = -10
        if xkp1[1] > 10:
            xkp1[1] = 10
        if xkp1[1] < -10:
            xkp1[1] = -10
        xk = xkp1
        if (xkp1[0] == 10 or xkp1[0] == -10) and (xkp1[1] == 10 or xkp1[1] == -10):
            break
        dgrad = calc_dgrad(xk)
        count_calc_func += m + 1
        norm_dgrad = norm(dgrad)
        k += 1
    return xk, count_calc_func

def global_search_1(eps : float = 0.1, l : int = 3):
    xmin = np.random.uniform(-10, 10, 2)
    xmin, count_calc_func = static_grad_medhod(xmin, 20, eps=eps)
    flag = 0
    fmin = func(xmin)
    count_calc_func += 1
    fk = fmin + 1
    while flag < l:
        xk = np.random.uniform(-10, 10, 2)
        xk, temp_calc_func = static_grad_medhod(xk, 20, eps=eps)
        fk = func(xk)
        count_calc_func += 1 + temp_calc_func
        if fk < fmin:
            fmin, xmin = fk, xk
            flag = 0
        else:
            flag += 1
    return xmin, count_calc_func
def global_search_2(eps : float = 0.1, l : int = 3):
    xmin, count_calc_func = static_grad_medhod(np.random.uniform(-10, 10, 2), 20, eps=eps)
    fmin = func(xmin)
    count_calc_func += 1
    flag = 0
    xk = xmin
    while flag < l:
        xk, temp_calc_func = simple_random_search(p=0.95, pe=0.01, xmin=xk)
        count_calc_func += temp_calc_func
        xk, temp_calc_func = static_grad_medhod(xk, 20, eps=eps)
        fk = func(xk)
        count_calc_func += temp_calc_func + 1
        if fk < fmin:
            fmin, xmin = fk, xk
            flag = 0
        else:
            flag += 1
    return xmin, count_calc_func

def global_search_3(x0 = None, eps : float = 0.1, m : int = 10, h : float = 0.1):
    if x0 is None:
        x0 = np.random.uniform(-10, 10, 2)
    elif len(x0) != 2 or m < 2:
        return 0
    
    gk = np.array([[0.0, 1.0]])
    for i in range(1, m):
        psi = radians(360*i/m)
        gk = np.concatenate((gk, np.array([[sin(psi), cos(psi)]])), axis=0)
    find_point = 0
    xmin, count_calc_func = static_grad_medhod(x0, 20, eps=eps)
    fmin = func(xmin)
    count_calc_func += 1
    while find_point < m:
        for i in range(m):
            xk, xkp1 = xmin, xmin + h * gk[i]
            fk, fkp1 = fmin, func(xkp1)
            count_calc_func += 1
            while xkp1[0] <= 10.0 and xkp1[0] >= -10.0 and xkp1[1] >= -10.0 and xkp1[1] <= 10.0 and fk < fkp1:
                xk = xkp1
                xkp1 += h * gk[i]
                fk, fkp1 = fkp1, func(xk)
                count_calc_func += 1
            if xkp1[0] <= 10.0 and xkp1[0] >= -10.0 and xkp1[1] >= -10.0 and xkp1[1] <= 10.0:
                xk, temp_calc_func = static_grad_medhod(xkp1, 20, eps=eps)
                fk = func(xk)
                count_calc_func += 1 + temp_calc_func
                if fk < fmin:
                    find_point = 0
                    xmin, fmin = xk, fk
                    break
                else:
                    find_point += 1
            else:
                find_point += 1
    return xmin, count_calc_func

def research_PSP():
    df = pd.DataFrame(columns=np.array(["Peps", "P","N", "(x, y)", "f(x, y)"]))
    P = [0.8, 0.9, 0.95, 0.99, 0.999]
    Pe = [0.1, 0.025, 0.01, 0.005, 0.001]
    for pe in Pe:
        for p in P:
            xmin = simple_random_search(p=p, pe=pe)
            df.loc[len(df) - 1] = np.array([f"{pe}", f"{p}", f"{find_N(p, pe)}", "({0:.3f}, {1:.3f})".format(xmin[0], xmin[1]), "{0:.5f}".format(-func(xmin))])
    out_exel(df, "PSP")

def research_global_search_1(M, seed=None):
    df = pd.DataFrame(columns=np.array(["m", "Count Calc Func", "(x, y)", "f(x, y)"]))
    for m in M:
        if not seed is None:
            np.random.seed(seed=seed)
        xmin, count_calc_func = global_search_1(eps=1e-3, l=m)
        df.loc[len(df) - 1] = np.array([str(m), str(count_calc_func), "({0:.3f}, {1:.3f})".format(*xmin), "{0:.5f}".format(-func(xmin))])
    return df

def research_global_search_2(M, seed=None):
    df = pd.DataFrame(columns=np.array(["m", "Count Calc Func", "(x, y)", "f(x, y)"]))
    for m in M:
        if not seed is None:
            np.random.seed(seed=seed)
        xmin, count_calc_func = global_search_2(eps=1e-3, l=m)
        df.loc[len(df) - 1] = np.array([str(m), str(count_calc_func), "({0:.3f}, {1:.3f})".format(*xmin), "{0:.5f}".format(-func(xmin))])
    return df

def research_global_search_3(M,seed=None):
    df = pd.DataFrame(columns=np.array(["m", "Count Calc Func", "(x, y)", "f(x, y)"]))
    for m in M:
        if not seed is None:
            np.random.seed(seed=seed)
        xmin, count_calc_func = global_search_3(eps=1e-6, m=m)
        df.loc[len(df) - 1] = np.array([str(m), str(count_calc_func), "({0:.3f}, {1:.3f})".format(*xmin), "{0:.5f}".format(-func(xmin))])
    return df

temp_M = np.array([4, 8, 10, 20, 30, 40], dtype=np.int8)

df1 = research_global_search_1(temp_M, seed=599)
df2 = research_global_search_2(temp_M, seed=599)
df3 = research_global_search_3(temp_M, seed=599)

if Name_OutFile != "":
        with pd.ExcelWriter(Name_OutFile + ".xlsx") as writer:
            df1.to_excel(writer, sheet_name="Algorithm 1", float_format="%.8f", index=False)
            df2.to_excel(writer, sheet_name="Algorithm 2", float_format="%.8f", index=False)
            df3.to_excel(writer, sheet_name="Algorithm 3", float_format="%.8f", index=False)
