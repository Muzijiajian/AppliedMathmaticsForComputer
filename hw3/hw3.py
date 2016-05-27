#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt

def gmm(X, K_or_centroids):
    '''
    Expectation Maximization iteration implementation of Gaussian Mixture Model
    :param X: input data, N-by-D data array
    :param K_or_centroids: either K indicating the number of components or a K-by-D matrix indicating the
                           choosing of the initial K centroids.
    :return:
    '''
    

totalPoints = 400
weights = [0.25, 0.25, 0.5]
mu1 = [4, -4]
cov1 = [[3, 0], [0, 3]]
mu2 = [2, 2]
cov2 = [[1.8, 0], [0, 1.5]]
mu3 = [-1, -1]
cov3 = [[1.5, 0], [0, 3.5]]
# 这里不使用转置的话将产生ValueError: too many values to unpack的错误,原因在于返回的参数将会是50*2即有50个参数而不是两个参数
x, y = np.random.multivariate_normal(mu1, cov1, 50).T
data1 = np.random.multivariate_normal(mu1, cov1, int(totalPoints*weights[0])).T
data2 = np.random.multivariate_normal(mu2, cov2, int(totalPoints*weights[1])).T
data3 = np.random.multivariate_normal(mu3, cov3, int(totalPoints*weights[2])).T
# print data1.shape
fig1 = plt.figure(num=1)
plt.plot(data1[0,:], data1[1,:], 'or', label='mu[4,-4] cov[3,3]')
plt.plot(data2[0,:], data2[1,:], 'og', label='mu[2,2] cov[1.8,1.5]')
plt.plot(data3[0,:], data3[1,:], 'ob', label='mu[-1,-1] cov[1.5,3.5]')
plt.axis('equal')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -.15), fancybox=True, ncol=3)
plt.show()
# fig1.savefig("Original data distribution")
mixed_data = data1.T.tolist() + data2.T.tolist() + data3.T.tolist()
mixed_data = np.array(mixed_data)
# fig2 = plt.figure(num=2)
# plt.plot(mixed_data[:,0], mixed_data[:,1], 'o')
