import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def read(end, lag):
    length = end - 1962 + 1
    data = pd.read_excel('data.xlsx')
    flag = (data['year'] >= 1962 - lag) & (data['year'] <= end)
    data = data[flag]
    t = data['year'].values[lag:]
    x = data['rainfall'].values
    y = data['runoff'].values[lag:]
    z = data['mining'].values[lag:]
    X = np.array([np.flip(x[i:i + lag]) for i in range(length)])
    X = sm.add_constant(X)
    Z = np.array([np.flip(z[i:i + lag + 1]) for i in range(length)])
    z_cum = np.array([np.sum(z[:i]) for i in range(length)]).reshape([-1, 1])
    print(Z)
    print(X.shape, z_cum.shape, Z.shape)
    return t, np.hstack([X, z_cum, Z]), y


def fit(lag):
    t, X, y, z = read(1979, lag)
    mod = sm.OLS(y, X)
    res = mod.fit()
    return res


def draw():
    lag = np.arange(1, 10)
    AIC = []
    BIC = []
    R2 = []
    R2_adj = []
    for l in lag:
        res = fit(l)
        AIC.append(res.aic)
        BIC.append(res.bic)
        R2.append(res.rsquared)
        R2_adj.append(res.rsquared_adj)

    plt.plot(lag, AIC, label='赤池信息量准则')
    plt.plot(lag, BIC, label='贝叶斯信息量准则')
    plt.xlabel(r'最大滞后年数 $L$')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.plot(lag, R2, label='决定系数')
    plt.xlabel(r'最大滞后年数 $L$')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    t, XZ, y = read(2020, 3)
    print(t)
    print(y)
    print(XZ)


if __name__ == '__main__':
    main()
