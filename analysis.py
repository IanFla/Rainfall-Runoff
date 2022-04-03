import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def read(end, lag):
    length = end - 1962 + 1
    data = pd.read_excel('data.xlsx')
    flag = (data['year'] >= 1962 - lag + 1) & (data['year'] <= end)
    data = data[flag]
    t = data['year'].values[lag - 1:]
    x = data['rainfall'].values
    y = data['runoff'].values[lag - 1:]
    X = np.array([x[i:i + lag] for i in range(length)])
    X = sm.add_constant(X)
    return t, X, y


def fit(lag):
    t, X, y = read(1979, lag)
    mod = sm.OLS(y, X)
    res = mod.fit()
    return res


def draw():
    lag = np.arange(2, 11)
    R2 = []
    R2_adj = []
    AIC = []
    BIC = []
    for l in lag:
        res = fit(l)
        R2.append(res.rsquared)
        R2_adj.append(res.rsquared_adj)
        AIC.append(res.aic)
        BIC.append(res.bic)

    plt.plot(lag, R2, label='决定系数')
    plt.plot(lag, R2_adj, label='经调整决定系数')
    plt.xlabel(r'滞后年数 $L$')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.plot(lag, AIC, label='赤池信息量')
    plt.plot(lag, BIC, label='贝叶斯信息量')
    plt.xlabel(r'滞后年数 $L$')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # draw()
    res = fit(4)
    t, X, y = read(2020, 4)
    res2 = res.get_prediction(X)
    conf = res2.conf_int()
    plt.plot(t, y, 'g', label='实际泉流量')
    plt.plot(t, res2.predicted_mean, 'r', label='理论泉流量')
    plt.plot(t, conf[:, 0], '--r')
    plt.plot(t, conf[:, 1], '--r')
    plt.xlabel('年份')
    plt.ylabel(r'泉流量（$m^3/s$）')
    plt.legend()
    plt.tight_layout()
    plt.show()

    data = np.hstack([y.reshape([-1, 1]), res2.predicted_mean.reshape([-1, 1]), conf])
    np.savetxt('test.csv', data, delimiter=',', fmt='%s')

    flag = (t >= 1980)
    var = np.sum(res2.var_pred_mean[flag]) / (np.sum(flag) ** 2)
    diff = np.sum((res2.predicted_mean - y)[flag]) / np.sum(flag)
    print(diff, [diff - 1.96 * np.sqrt(var), diff + 1.96 * np.sqrt(var)])


if __name__ == '__main__':
    main()