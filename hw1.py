#encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.stats import norm

def curve_fit_weight(train_data, train_label, degree, reg=0):
    '''
    This curve fit will use The Normal Equations
    当feature数目大于1e5时,使用Gradient Descent
    degree: The degree of polynomial
    reg: The regularization part
    '''
    data_size = train_data.size
    data = np.matrix(np.zeros(shape=(degree+1, data_size)))
    # 根据多项式的degree分别计算此时数据经过假设函数后的值, 每一列为一条数据
    for i in range(0, degree+1):
        value = train_data**(degree-i)
        data[i, :] = value
    # Using normal equations
    weights = LA.inv(data*data.T + reg*np.eye(degree+1)) * data * np.matrix(train_label).T
    return weights.A1

numpoints = 9   #数据点的个数
# 绘制坐标轴定义
fig = plt.figure(num=1)
plt.xlabel('x')
plt.ylabel('t')
plt.xlim(-.2, 1.2)
plt.ylim(-1.5, 1.5)
# 绘制标准的sin曲线,用散点图逼近的方式
x1 = np.linspace(0, 1, 1e4)
y1 = np.sin(2 * x1 * np.pi)
plt.plot(x1, y1, color='green', label='Sin(x)', linewidth=2)
# 绘制原始数据点
x2 = np.linspace(0, 1, numpoints)  #np.random.normal(size=numpoints)
noise1 = norm.rvs(0, size=numpoints, scale=0.15)
noise2 = np.random.normal(0, 0.2, size=numpoints)
y2 = np.sin(x2 * 2*np.pi) + noise1
plt.plot(x2, y2, 'o', label='data with noise', mew=2, mec='b', mfc='none', ms=6)
# 绘制多项式逼近曲线
Reg = np.exp(-18)
degree = 9
weight = curve_fit_weight(x2, y2, degree, Reg)
print 'Fitting Parameters:', weight
po = np.poly1d(weight)          # 产生一个参数为weight的多项式
new_x = np.linspace(0, 1, 1e4)
plt.plot(new_x, po(new_x), '-r', label='Poly Fitting Line(degree=9, lnR=e-18)', linewidth=2)
# 绘制图例说明,图例显示在图表的上方
plt.legend(loc='lower center', bbox_to_anchor=(0.5, .95), fancybox=True, ncol=3)
# 绘制后保存图
plt.show()
fig.savefig('poly_degree_9.png')