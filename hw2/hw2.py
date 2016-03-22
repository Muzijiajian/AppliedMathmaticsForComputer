#encoding=utf-8
# 这个python file主要用于实现PCA(主成成分分析)的算法实现以及可视化
from __future__ import division
__author__ = 'Dejian,Li 11521046'
__date__ = '$2016-3-19$'

import numpy as np
import matplotlib.pyplot as plt
import sys

def prepare_data(fname):
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
        tmplist = line.rstrip().split(',')
        # line = line.split(',')
        labels.append(tmplist[-1])
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
    samples = np.asarray(samples).T
    labels = np.asarray(labels).T
        # print type(labels)
        # print np.shape(labels)
        # print np.shape(samples)
        # print labels
    return (samples, labels)

def normalize(data):
    '''
    对向量模长进行归一化,这里采用的是Z-score的方法
    :param data: 输入需要归一化的向量
    :return: 归一化后的向量
    '''
    # 0 stands for do it in column, 1 stands for row
    mu = np.mean(data.astype(np.float), axis=0)
    print type(mu)
    sigma = np.std(data.astype(np.float), axis=0)
    print type(sigma)
    norm_data = (data - mu) / sigma
    return norm_data

def my_pca(dataMat, topNfeat=999999):
    '''
    使用pca算法是对数据进行主成员提取, 这里利用eig分解的方式
    :param dataMat: features*numCases
    :param topNfeat: 应用的n个特征向量
    :return: 降维后的数据集lowDataMat, eigVals & eigVecs
    '''
    meanVals =np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    scatter_matrix = np.cov(meanRemoved, rowvar=0)
    eigVals, eigVecs = np.linalg.eig(mat(scatter_matrix))
    #
    print eigVals[:,2]
    print eigVecs[:,2]


(train_data, train_label) = prepare_data('optdigits.tra')
# train_data = normalize(train_data)
# print type(train_data)
my_pca(train_data, topNfeat=2)
(test_data, test_label) = prepare_data('optdigits.tes')
print np.shape(train_data)