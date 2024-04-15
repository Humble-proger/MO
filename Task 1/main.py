from math import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def f(x):
    return (x-2)**2

def delta(eps):
    calc_delta = eps / 2
    if calc_delta < eps:
        return calc_delta
    else:
        raise "delta считается не правильно"
def dichotomy(start, end, eps):
    a, b, iter, func_calc = start, end, 0, 0
    df = pd.DataFrame(columns=np.array(['x1','x2', 'f(x1)', 'f(x2)','a', 'b', 'len', 'prev/now']))
    while abs(b-a) > eps:
        a_step, b_step = a, b
        x1 = (a + b - delta(eps)) / 2
        x2 = (a + b + delta(eps)) / 2
        if f(x1) > f(x2):
            a = x1
        else:
            b = x2
        iter += 1
        func_calc += 2
        df.loc[len(df)] = np.array([x1, x2, f(x1), f(x2), a, b, b-a, (b_step - a_step)/(b-a)])
    #print(f'Минимальный отрезок: [{a}, {b}]')
    return df, func_calc

def golden_section(start, end, eps):
    a, b, iter, func_calc = start, end, 0, 2
    df = pd.DataFrame(columns=np.array(['x1','x2', 'f(x1)', 'f(x2)', 'a', 'b', 'len', 'prev/now']))
    val = (sqrt(5) - 1)/2
    x1 = a + (1 - val) * (b - a)
    x2 = a + val * (b - a)
    f1, f2 = f(x1), f(x2)
    while abs(a - b) > eps:
        a_step, b_step = a, b
        if f1 > f2:
            a = x1
            x1, x2 = x2, a + val * (b - a)
            f1, f2 = f2, f(x2)
        else:
            b = x2
            x2, x1 = x1, a + (1 - val) * (b - a)
            f2, f1 = f1, f(x1)
        iter += 1
        func_calc += 1
        df.loc[len(df)] = np.array([x1, x2, f(x1), f(x2), a, b, b-a, (b_step - a_step)/(b-a)])
    return df, func_calc


def find_n(a, b, eps):
    n = 0
    while  (b - a) / eps >= Fib_num(n):
        n += 1
    return n

def Fib_num(n):
    val = sqrt(5)
    return (((1 + val)/2)**n - ((1-val)/2)**2) / val

def Fib_method(start, end, eps):
    n = find_n(start, end, eps)
    l = start + Fib_num(n-2) * (end - start)/Fib_num(n)
    m = start + Fib_num(n-1) * (end - start)/Fib_num(n)
    a, b, k, calc_func = start, end, 0, 2
    df = pd.DataFrame(columns=np.array(['x1','x2', 'f(x1)', 'f(x2)','a', 'b', 'len', 'prev/now']))
    func_l, func_m = f(l), f(m)
    while k != n - 3:
        a_step, b_step = a, b
        if func_l > func_m:
            a = l
            l = m
            m = a + Fib_num(n - k - 2) * (b - a) / Fib_num(n - k - 1)
            func_l = func_m
            func_m = f(m)
        else:
            b = m
            m = l
            l = a + Fib_num(n - k - 3) * (b - a) / Fib_num(n - k - 1)
            func_m = func_l
            func_l = f(l)
        k += 1
        calc_func += 1
        df.loc[len(df)] = np.array([l, m, f(l), f(m), a, b, b-a, (b_step - a_step)/(b-a)])
    m = l + eps
    func_m = f(m)
    calc_func += 1
    if func_l >= func_m:
        a = l
    else:
        b = m
    #print(f'Минимальный отрезок: [{a}, {b}]')
    df.loc[len(df)] = np.array([l, m, f(l), f(m), a, b, b-a, (b_step - a_step)/(b-a)])
    return df, calc_func

def find_direction(x, delta):
    if f(x) > f(x + delta):
        return delta
    return -delta

def find_min_interval(x, delta):
    h = 2 * find_direction(x, delta)
    xk, k, func_calc = x + h, 0, 0
    df = pd.DataFrame(columns=np.array(['xi', 'f(xi)']))
    
    while f(x) > f(xk):
        x, xk = xk, xk + h
        k += 1
        func_calc += 2
        df.loc[len(df)] = np.array([x, f(x)])
    #print(f'Минимальный отрезок: [{x - h}, {xk}], кол-во итераций: {k}')
    return df

def grafic():
    a, b = -2, 20
    axis_x = np.array([np.log10(eps) for eps in [10**(-i) for i in reversed(range(1, 8))]])
    axis_y_dec = np.array([dichotomy(a, b, eps)[1] for eps in [10**(-i) for i in reversed(range(1, 8))]])
    axis_y_gold = np.array([golden_section(a, b, eps)[1] for eps in [10**(-i) for i in reversed(range(1, 8))]])
    axis_y_fib = np.array([Fib_method(a, b, eps)[1] for eps in [10**(-i) for i in reversed(range(1, 8))]])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor('#2D3332')
    ax.plot(axis_x, axis_y_dec,'o-', label='Dichotomy', color='#E3B414')
    ax.plot(axis_x, axis_y_gold,'o-', label='Golden Section', color='#DD14E3')
    ax.plot(axis_x, axis_y_fib,'o-', label="Fibonacci Method", color='red')
    ax.set_xlabel("Log(eps)")
    ax.set_ylabel("Count Calc Function")
    ax.grid(color='#14E39E')
    ax.spines['right'].set_color('#14E39E')
    ax.spines['top'].set_color('#14E39E')
    ax.spines['bottom'].set_color('#14E39E')
    ax.spines['left'].set_color('#14E39E')
    ax.xaxis.label.set_color('white')
    ax.tick_params(colors='#14E39E')
    ax.yaxis.label.set_color('white')
    ax.legend(facecolor='#2D3332', labelcolor='white')
    fig.set_facecolor('#2D3332')
    plt.title("Optimization methods", color='white')
    plt.show()

if __name__ == "__main__":
    grafic()
    with pd.ExcelWriter("OutputFile.xlsx") as writer:
        dichotomy(-2, 20, 1e-7)[0].to_excel(writer, sheet_name="Dichotomy Method", index_label='i', float_format='%.8f')
        golden_section(-2, 20, 1e-7)[0].to_excel(writer, sheet_name="Golden Section Method", index_label='i', float_format='%.8f')
        Fib_method(-2, 20, 1e-7)[0].to_excel(writer, sheet_name="Fibonacci Method", index_label='i', float_format='%.8f')
        find_min_interval(20, 0.5).to_excel(writer, sheet_name="Minimum interval", float_format='%.2f', index_label='i')