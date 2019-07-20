# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:38:42 2019
本程序利用主成分分析方法对鸢尾花数据矩阵进行降维
参考资料：《深度学习》，2.12实例：主成分分析
        周志华《机器学习》，10.3主成分分析
        https://github.com/zhaoxingfeng/PCA
@author: yanji
"""

import numpy as np
import pandas as pd

def PCAcomponent(dataX,dimL):
    """
    function:将数据矩阵dataX降到dimL维
    Parameters:dataX - 数据矩阵
               dimL - 数据目标维度
    Returns: dataLowDim - 降维后的数据矩阵
             variance_ratio - 降维后各个维度所占的方差百分比
             small_eigVect - 对样本数据进行缩放的投影矩阵
    """
    dataMat=dataX-np.mean(dataX,axis=0) #对原始数据矩阵进行中心化
    XTX=dataMat.T*dataMat #计算协方差矩阵X.T*X
    eigVal,eigVect=np.linalg.eig(XTX) #计算协方差矩阵的特征值与特征向量
    eigValInd=np.argsort(-eigVal) #按照特征值从大到小顺序得到索引值列表
    small_eigVect=eigVect[eigValInd[0:dimL],:].T #取前dimL个最大的特征值对应的特征向量组成投影矩阵
    dataLowDim=dataMat*small_eigVect #投影得到降维后的新矩阵
    variance_ratio=[eigVal[i]/sum(eigVal) for i in eigValInd[0:dimL]] #计算降维后各个维度所占的方差百分比
    return dataLowDim,variance_ratio,small_eigVect  #返回降维后的矩阵，及各维度的方差百分比,投影矩阵
    



if __name__=='__main__':
    irisData=pd.read_csv(r'iris.txt',header=None)  #读取原始数据
    irisMat=np.mat(irisData.iloc[:,0:-1])     #提取样本数据矩阵
    irisMatDimL,iris_variance_ratio,projectionMat=PCAcomponent(irisMat,3)  #利用主成分分析法对样本数据矩阵进行降维
    