#encoding=utf-8
# 这个python file主要用于实现PCA(主成成分分析)的算法实现以及可视化
from __future__ import division
__author__ = 'Dejian,Li 11521046'
__date__ = '$2016-3-19$'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  #用于显示灰度图像
import sys

def prepare_data(fname, chooseDigit):
    '''
    读取http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    网站的数据,使用的是已处理的版本,其数据格式为
    All input attributes are integers in the range 0..16
    The last attributes is the class code 0..9
    原始图片32*32的图片里的4*4(转变为每一行64+1个数字)的块组成数据一行的数值,因此每一个数值的范围为0-16
    :param fname: 需要读取的文件名字
    :return: 整个数据集,samples & labels
    '''
    samples = []
    labels = []
    i = 0
    try:
        handle = open(fname, 'r')
    except:
        print 'Something wrong happened, file can not be opened.\n'
    for line in handle:
        tmplist = map(int, line.rstrip().split(','))
        # line = line.split(',')
        if tmplist[-1] == chooseDigit:
        # labels.append(tmplist[-1])
            tmplist = tmplist[:-1]
            samples.append(tmplist)
        # i = i+1
        # print labels
        # print samples
        # if i==2:
        #     samples = np.asarray(samples)
        #     print type(samples)
        #     break
    # 将生成的list转变成numpy array
    samples = np.mat(samples).T
    labels = np.mat(labels).T
        # print type(labels)
        # print np.shape(labels)
        # print np.shape(samples)
        # print labels
    return (samples, labels)

def normalize(dataMat):
    '''
    对向量模长进行归一化,这里采用的是Z-score的方法
    :param data: 输入需要归一化的向量
    :return: 归一化后的向量, 其实就是减去向量的均值
    '''
    # 0 stands for do it in column, 1 stands for row
    EPSILON = 10e-4
    mu = np.mean(dataMat, axis=1)
    print type(mu)
    sigma = np.std(dataMat, axis=1)
    # print type(sigma)
    norm_data = (dataMat - mu) / (sigma+EPSILON)
    return norm_data

def my_pca(dataMat, topNfeat=999999):
    '''
    使用pca算法是对数据进行主成员提取, 这里利用eig分解的方式
    :param dataMat: features*numCases
    :param topNfeat: 应用的n个特征向量
    :return: 降维后的数据集lowDataMat, eigVals & eigVecs
    '''
    scatter_matrix = dataMat * dataMat.T
    print 'Type of scatter_matrix', type(scatter_matrix)
    eigVals, eigVecs = np.linalg.eig(scatter_matrix)
    # 将eigVals对角矩阵化回来
    # eigVals = np.linalg.diag(eigVals)
    print 'Shape of eigValues', np.shape(eigVals)
    print 'Shape of eigVectors', np.shape(eigVecs)
    print eigVals[0:topNfeat]
    # print eigVecs[:, 1].T
    return eigVals[0:topNfeat], eigVecs[:, 0:topNfeat]

def plot_eigVecs(eigVecs):
    '''
    imshow函数表示绘制二维图像 cmap=plt.cm.gray(画布颜色定义为灰色)
    :param eigVecs: 主成成分特征向量
    :return: None
    '''
    figure1 = plt.figure(num=1)
    plt.subplot(121)
    plt.xlabel('First Component')
    plt.title('$\lambda = %f$' % eigValues[0])
    plt.imshow(eigVecs[:,0].reshape(8,8), cmap=cm.gray)
    plt.subplot(122)
    plt.title('$\lambda = %f$' % eigValues[1])
    plt.xlabel('Second Component')
    plt.imshow(eigVecs[:,1].reshape(8,8), cmap=cm.gray)
def rotate_Data(dataMat, eigVecs):
    '''
    计算经过主成成分特征向量进行缩放后的数据在新空间下的数值
    :param dataMat: features*numCases
    :param eigValues: 主要成分的特征值
    :param eigVecs: 主要成分的特征向量
    :return: 新空间下的数据集, features*numCases
    '''
    xRot = eigVecs.T * dataMat
    # newData = eigVecs * eigVecs.T * dataMat
    # print newData
    print 'Shape of rotate data', np.shape(xRot)
    print 'Type of rotate data', type(xRot)
    return xRot

def plotData(dataMat):
    '''
    根据data数据集格式进行在二维坐标轴上进行可视化操作
    :param dataMat: features*numCases
    :return: None
    '''
    figure2 = plt.figure(num=2)
    # print dataMat[1].size
    # print dataMat[:, 388][1]
    # 将数据集拆分为x,y两个ndarray进行可视化处理
    xlist = []
    ylist = []
    for i in range(dataMat[0].size):
        x = int(dataMat[:, i][0]) / 5
        y = int(dataMat[:, i][1]) / 5
        xlist.append(x)
        ylist.append(y)

    print 'Length of list x and list y', len(xlist), len(ylist)
    maxX = max(xlist)
    maxY = max(ylist)
    # print 'Max of X and Y is', maxX, maxY
    xlist = np.array(xlist)   # 这里因为数据点过于分散,因此同时进行缩放
    ylist = np.array(ylist)
    plt.plot(xlist, ylist, 'o', color='green')
    plt.xlabel('First Component')
    plt.ylabel('Second Component')
    plt.xlim(-6, 6)
    plt.ylim(-8, 8)
    # 创建网格
    plt.grid()

(train_data, train_label) = prepare_data('optdigits.tra', chooseDigit=3)
original_train_data = train_data
norm_data = normalize(train_data)
# print 'One samples of training data', train_data[:, 0].T
# print 'Labels', train_label[:].T
print 'Type of training data', type(norm_data)
print 'Shape of training data', np.shape(norm_data)
print 'Type of training label', type(train_label)
print 'Shape of training label', np.shape(train_label)
# print original_train_data
(eigValues, eigVecs) = my_pca(norm_data, topNfeat=2)
# 计算经过主成成分提取后所形成的新数据集
projData = rotate_Data(train_data, eigVecs)
# 绘制主成成分数字情形
plot_eigVecs(eigVecs)
# 在坐标轴上绘制只是提取两个主要成分的数据
plotData(projData)

# (test_data, test_label) = prepare_data('optdigits.tes')
plt.show()