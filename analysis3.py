import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def read(end, lag):
    length = end - 1962 + 1
    data = pd.read_excel('data.xlsx')
    flag = (data['year'] >= 1962 - 3) & (data['year'] <= end)
    data = data[flag]
    t = data['year'].values[3:]
    x = data['rainfall'].values
    y = data['runoff'].values[3:]
    X = np.array([np.flip(x[i:i + 3]) for i in range(length)])
    X = sm.add_constant(X)

    data2 = pd.read_excel('data.xlsx')
    flag2 = (data2['year'] >= 1962 - lag) & (data2['year'] <= end)
    data2 = data2[flag2]
    z = data2['mining'].values
    Z = np.array([np.flip(z[i:i + lag + 1]) for i in range(length)])
    z_cum = np.array([np.sum(z[:i]) for i in range(length)]).reshape([-1, 1])
    return t, np.hstack([X, z_cum, Z]), y


def fit(lag):
    t, XZ, y = read(2020, lag)
    mod = sm.OLS(y, XZ)
    res = mod.fit()
    return res


def draw():
    lag = np.arange(0, 10)
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
    # draw()
    res = fit(0)
    print(res.summary())

    t, XZ, y = read(2020, lag=0)
    res2 = res.get_prediction(XZ)
    sigma2 = np.sum((y - res2.predicted_mean) ** 2) / (y.size - XZ.shape[1])

    plt.plot(t, y, 'g', label='实际泉流量')
    plt.plot(t, res2.predicted_mean, 'r', label='理论泉流量')
    plt.plot(t, res2.predicted_mean + 1.96 * np.sqrt(sigma2), '--r')
    plt.plot(t, res2.predicted_mean - 1.96 * np.sqrt(sigma2), '--r')
    plt.xlabel('年份')
    plt.ylabel(r'泉流量（$m^3/s$）')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
