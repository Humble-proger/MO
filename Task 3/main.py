import numpy as np
from math import sqrt, log
import matplotlib.pyplot as plt
import pandas as pd

track_method = np.array([])
Name_OutFile = "OutputFile"

def penalty_function(point):
        r0 = 4
        def g1(point):
            return -point[0] - point[1] + 2
        return r0 * abs(g1(point))
def f(point):
    def objective_function(point):  
        return 10*(point[1]-point[0])**2 + point[1]**2
    return objective_function(point) + penalty_function(point)

def fg(point):
    def objective_function(point):  
        return 10*(point[1]-point[0])**2 + point[1]**2
    return objective_function(point)
def matrix_to_string(vector, flag=False):
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
            out += "{0:.3f}".format(vector[i]) + " "
        out += "]\n"
    return out

def Nelder_Mid(func, x0, eps, alpha : float = 1, beta : float = 0.5, gamma : float = 2, t : float = 1, max_iter : int = 100, _table : bool = False, _track : bool = True):
    def norm(f_res, f_min):
        return sqrt(sum([(f_res[i] - f_min)**2 for i in range(len(f_res))]))
    if _table:
        df = pd.DataFrame(columns=np.array(["xmin", 'fmin(x, y)', '|fi - fm1|', 'count calc func', 'penalty function']))
    N = len(x0)
    count_iter = 0
    count_calc_func = 0
    if N < 2:
        return -1
    D = np.zeros((N, N+1))

    d1 = t*(sqrt(N+1) + N - 1)/(N * sqrt(2))
    d2 = t * (sqrt(N + 1) - 1)/(N*sqrt(2))
    for i in range(N):
        D[i, 0] = x0[i]
        for i in range(N):
            D[i, i+1] = x0[i] + d1
            for j in range(i+2, N+1):
                D[i, j] = x0[i] + d2
                D[j - 1, i + 1] = x0[j - 1] + d2
    if _track:
        global track_method
        track_method = np.array([D])
    res_f = np.array([func(D[:, i]) for i in range(N+1)])
    count_calc_func += N+1
    func_c = 0
    index_min = 0
    if _table:
        old_func = func(x0)
        df.loc[len(df)] = np.array([matrix_to_string(x0), old_func, 0, count_calc_func, penalty_function(x0)])
    while norm(res_f, func_c) >= eps and count_iter < max_iter:
        func_c = min(res_f)
        index_max = np.where(res_f == max(res_f))[0][0]
        index_min = np.where(res_f == min(res_f))[0][0]
        temp_res = sorted(res_f)
        
        xc = np.zeros((N,))
        for i in range(N + 1):
            if i != index_max:
                xc += D[:, i]
        xc /= N
        xr = (1 + alpha)*xc - alpha*D[:, index_max]
        f_r = func(xr)
        count_calc_func += 1
        if f_r < temp_res[0]:
            xe = (1 - gamma)*xc + gamma*xr
            f_e = func(xe)
            count_calc_func += 1
            if f_e < f_r:
                D[:, index_max] = xe
                res_f[index_max] = f_e
            else:
                D[:, index_max] = xr
                res_f[index_max] = f_r
        elif f_r < temp_res[1]:
            D[:, index_max] = xr
            res_f[index_max] = f_r
        else:
            if f_r > res_f[N]:
                D[:, index_max] = xr
                temp_res[N] = f_r
            xs = beta*D[:, index_max] + (1-beta)*xc
            f_s = func(xs)
            count_calc_func += 1
            if f_s < temp_res[N]:
                D[:, index_max] = xs
                res_f[index_max] = f_s
            else:
                xl = D[:, index_min]
                for i in range(N+1):
                    if i != index_min:
                        D[:, i] = xl + (D[:, i] - xl)
                        res_f[i] = func(D[:, i])
                        count_calc_func += 1
        count_iter += 1
        if _table:
            df.loc[len(df)] = np.array([matrix_to_string(D[:, index_min]), func_c ,abs(func_c - old_func), count_calc_func, penalty_function(D[:, index_min])])
            old_func = func_c
        if _track:
            track_method = np.vstack((track_method, [D]))
    index_min = np.where(res_f == min(res_f))[0][0]
    if _table:
        df.loc[len(df)] = np.array([matrix_to_string(D[:, index_min]), res_f[index_min], abs(res_f[index_min] - old_func), count_calc_func, penalty_function(D[:, index_min])])
        return df, D[:, index_min]
    return D[:, index_min]

def create_grafic_track(res):
    x_grid = np.arange(-15, 10, 0.5)
    y_grid = np.arange(-15, 10, 0.5)
    x, y = np.meshgrid(x_grid, y_grid)
    func_z = 10 * (y - x) ** 2 + y ** 2
    cs = plt.contour(x, y, func_z, zorder=1, levels=[10, 25, 50, 100, 150, 300, 600, 1200, 2400, 4200])
    plt.clabel(cs)
    if len(track_method) > 0:
        for i in range(len(track_method)):
            temp_x, temp_y = track_method[i][0], track_method[i][1]
            plt.plot(np.append(temp_x, temp_x[0]), np.append(temp_y, temp_y[0]), zorder=2)
        plt.scatter(res[0], res[1], zorder=3)
    plt.title("Nelder Mid Method")

    plt.show()

def out_exel(df):
    if Name_OutFile != "":
        with pd.ExcelWriter(Name_OutFile + ".xlsx") as writer:
            df.to_excel(writer, sheet_name="Nelder Mid Method", index_label='i', float_format="%.8f")

def main():
    df, res = Nelder_Mid(f, np.array([-10, -5]), 1e-2, _table=True, max_iter=500)
    out_exel(df)
    print(res, len(df), fg(res), fg(res) - 0.975609756, sum(res), abs(penalty_function(res)))
    create_grafic_track(res)


if __name__ == "__main__":
    main()
