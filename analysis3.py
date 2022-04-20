import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def read():
    data = pd.read_excel('data.xlsx')
    flag = (data['year'] >= 1962) & (data['year'] <= 2020)
    data = data[flag]
    t = data['year'].values
    x = data['mining'].values
    X = np.array([[1.0, x[i], np.sum(x[:i])] for i in range(x.size)])
    y = data['loss'].values
    flag = (t >= 1980)
    return t[flag], X[flag], y[flag]


def main():
    t, X, y = read()
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(res.summary())
    res2 = res.get_prediction(X)

    plt.plot(t, y, 'g', label='理论损失泉流量')
    plt.plot(t, res2.predicted_mean, 'r', label='预测损失泉流量')
    plt.xlabel('年份')
    plt.ylabel(r'损失泉流量（$m^3/s$）')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
