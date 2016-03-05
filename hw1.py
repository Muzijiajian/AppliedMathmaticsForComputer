#encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt

numpoints = 10
numPloy = 3

# 绘制标准的sin曲线,用散点图逼近的方式
x1 = np.linspace(0, 1, 1e4)
x2 = (2 * np.linspace(0, 1, 10) + np.random.normal(size=numpoints)) * np.pi
y1 = np.sin(2 * x1 * np.pi)
y2 = np.sin(x2)
f1 = plt.figure(num=1)
plt.plot(x1, y1, color='green', linewidth=2)
# plt.scatter(x2, y2)
plt.plot()
plt.xlabel('x')
plt.ylabel('t')
plt.xlim(0, 1)
plt.ylim(-1.5, 1.5)
# plt.title('M = 3')
print(x2)

plt.legend()
plt.show()