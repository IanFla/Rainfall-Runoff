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
    return t, X, y, z


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
    # draw()
    res = fit(3)
    print(res.summary())

    t, X, y, z = read(2021, 3)
    res2 = res.get_prediction(X)
    flag = (t >= 1962) & (t <= 1979)
    sigma2 = np.sum((y[flag] - res2.predicted_mean[flag]) ** 2) / (flag.sum() - X.shape[1])

    plt.plot(t, y, 'g', label='实际泉流量')
    plt.plot(t, res2.predicted_mean, 'r', label='理论泉流量')
    plt.plot(t, res2.predicted_mean + 1.96 * np.sqrt(sigma2), '--r')
    plt.plot(t, res2.predicted_mean - 1.96 * np.sqrt(sigma2), '--r')
    plt.xlabel('年份')
    plt.ylabel(r'泉流量（$m^3/s$）')
    plt.legend()
    plt.tight_layout()
    plt.show()

    flag = (t >= 1980)
    plt.plot(t[flag], res2.predicted_mean[flag] - y[flag], 'b')
    plt.plot(t[flag], res2.predicted_mean[flag] + 1.96 * np.sqrt(sigma2) - y[flag], '--b')
    plt.plot(t[flag], res2.predicted_mean[flag] - 1.96 * np.sqrt(sigma2) - y[flag], '--b')
    plt.xlabel('年份')
    plt.ylabel(r'损失泉流量（$m^3/s$）')
    plt.tight_layout()
    plt.show()

    # data = np.hstack([t.reshape([-1, 1]), y.reshape([-1, 1]), res2.predicted_mean.reshape([-1, 1]), conf])
    # np.savetxt('test.csv', data, delimiter=',', fmt='%s')

    # flag1 = (t >= 1980) & (t <= 2000)
    # flag2 = (t >= 2000) & (t <= 2021)
    # flag = (t >= 1980)
    # var1 = np.sum(res2.var_pred_mean[flag1]) / (np.sum(flag1) ** 2)
    # diff1 = np.sum((res2.predicted_mean - y)[flag1]) / np.sum(flag1)
    # print(diff1, [diff1 - 1.96 * np.sqrt(var1), diff1 + 1.96 * np.sqrt(var1)])
    # var2 = np.sum(res2.var_pred_mean[flag2]) / (np.sum(flag2) ** 2)
    # diff2 = np.sum((res2.predicted_mean - y)[flag2]) / np.sum(flag2)
    # print(diff2, [diff2 - 1.96 * np.sqrt(var2), diff2 + 1.96 * np.sqrt(var2)])
    # var = np.sum(res2.var_pred_mean[flag]) / (np.sum(flag) ** 2)
    # diff = np.sum((res2.predicted_mean - y)[flag]) / np.sum(flag)
    # print(diff, [diff - 1.96 * np.sqrt(var), diff + 1.96 * np.sqrt(var)])


if __name__ == '__main__':
    main()
