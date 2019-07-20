# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:28:20 2019
本程序实现原型聚类中的K均值聚类算法
数据集为周志华《机器学习》中的表9.1西瓜数据集4.0
参考资料：周志华《机器学习》9.1～9.4节
@author: yanji
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def dataLoad(fileName):
    """
    function: 读取数据集，将数据结果保存到数组中并返回
    Parameters: fileName - 数据集文件名称，本次数据集为西瓜数据集4.0
    Returns: dataSet - 数据集（numpy数组形式）
    """
    dfData=pd.read_csv(fileName,encoding='gbk')#读取西瓜数据集4.0
    dataSet=np.array(dfData) #将dataframe数据集转为numpy数组形式
    return dataSet #返回数据集

def distanceCalc(ui,xj):
    """
    function:计算样本xj与均值向量ui的距离
    Parameters: ui - 均值向量
                xj - 数据样本
    Returns: dist - 距离计算结果
    """
    dist = math.sqrt(sum(pow(ui-xj,2))) #计算样本与均值向量的距离
    return dist

def Kmeans(dataSet,meanVec,K,maxIter,minChange):
    """
    function: 采用K均值算法对数据集进行聚类，将其分成K类
    Parameters: dataSet - 进行聚类的原始数据集
                meanVec - K个类别均值向量的初始化值
                K - 聚类类别数
                maxIter - 最大迭代轮数，若超过阈值，将停止迭代
                minChange - 最小调整幅度阈值，若分类调整幅度低于阈值，将停止迭代
    Returns:  meanVector - K个类别的均值向量
              classMark - 每个数据样本对应的类别标签，与meanVector对应，例如类别标签为1，则对应meanVector第一个类别，标签为K，则对应第K个类别
    """
    dataLen=len(dataSet)  #数据集样本个数
    """
    由于观察聚类结果随迭代轮数变化情况，则随机选取的初始化均值向量应该保持一致，应该在调用该函数前给出
    meanVector=np.zeros((K,dataSet.shape[1]),np.float)   #初始化均值向量
    meanVectorIndex=[] #初始化均值向量的索引
    for i in range(K):  #从数据集中随机抽取K个不同的样本保存到均值向量中
        nowIndex=round(random.uniform(0,dataLen-0.5))  #随机索引值
        while nowIndex in meanVectorIndex:   #如果该索引已经被抽取过，则重新抽取
            nowIndex=round(random.uniform(0,dataLen-0.5))
        meanVectorIndex.append(nowIndex) #更新索引值列表
        meanVector[i]=dataSet[nowIndex] #将该索引对应的样本添加到均值向量列表中
    """
    meanVector=meanVec.copy() #深拷贝方式复制初始化的均值向量
    classMark=np.zeros(dataLen,np.int)  #初始化数据集中所有样本的类别标签为0
    nowIter=0  #初始化迭代次数
    nowChange=dataLen #初始化一轮迭代中，数据集的分类调整幅度
    while(nowIter<maxIter and nowChange>=minChange): #当迭代次数超过阈值或者分类调整幅度小于阈值时退出循环
        nowChange=0  #初始化本轮迭代中，数据集的分类调整幅度为0
        for dataIndex in range(dataLen):  #遍历每个数据
            nowMark=0   #初始化类别标签
            nowDist=9999  #初始化距离度量
            for i in range(K):  #遍历均值向量
                distCalc=distanceCalc(meanVector[i],dataSet[dataIndex])  #计算该数据样本与均值向量的距离
                if distCalc<nowDist: #选择距离最小的均值向量所在的类为该数据的类别标签
                    nowDist=distCalc
                    nowMark=i+1
            if nowMark!= classMark[dataIndex]: #如果该数据的类别标签与之前不同，则对该数据类别进行更新，调整幅度加1
                nowChange+=1
                classMark[dataIndex]=nowMark
        for meanIndex in range(K):  #遍历每一个类别，对该类别的均值向量进行更新
            nowMean=np.zeros((1,dataSet.shape[1]),np.float) #初始化该类别均值向量
            numCount=0 #初始化该类别数据个数
            for dataIndex in range(dataLen):  #遍历数据集
                if classMark[dataIndex]==meanIndex+1: #在重新分好类的数据集中，寻找该类别的数据样本
                    nowMean+=dataSet[dataIndex] #对该类的数据样本值求和
                    numCount+=1
            if numCount>0: #如果该类存在标记的样本，则对该类均值向量进行更新
                meanVector[meanIndex]=nowMean/numCount    
        nowIter+=1  #迭代轮数加1
    return meanVector,classMark  #返回均值向量以及各样本数据分类结果
    
if __name__=='__main__':
    dataSet=dataLoad('watermelon40.csv')  #读取西瓜数据集4.0
    K=3 #设置分类类别数目
    meanVector=np.zeros((K,dataSet.shape[1]),np.float)  #初始化均值向量
    meanVectorIndex=[] #初始化均值向量的索引
    for i in range(K):  #从数据集中随机抽取K个不同的样本保存到均值向量中
        nowIndex=round(random.uniform(0,len(dataSet)-0.5))  #随机索引值
        while nowIndex in meanVectorIndex:   #如果该索引已经被抽取过，则重新抽取
            nowIndex=round(random.uniform(0,len(dataSet)-0.5))
        meanVectorIndex.append(nowIndex) #更新索引值列表
        meanVector[i]=dataSet[nowIndex] #将该索引对应的样本添加到均值向量列表中
    numIterList=[2,3,4,5]  #设置最大迭代次数列表
    meanVectorList=[]  #初始化不同最大迭代次数对应的均值向量列表
    classMarkList=[]   #初始化不同最大迭代次数对应的分类结果列表
    for numIter in numIterList:  #遍历最大迭代次数列表，分别进行K均值聚类
        nowmeanVector,nowclassMark=Kmeans(dataSet,meanVector,K,numIter,0)
        meanVectorList.append(nowmeanVector) #更新均值向量列表
        classMarkList.append(nowclassMark)  #更新分类结果列表
    lenNumIter=len(numIterList)  #迭代次数列表的数据个数
    markers=['s','x','o']   #三种标记类型
    colors=['red','blue','lightgreen']  #三种颜色
    fig,axs = plt.subplots(1,4,figsize=(16,4))  #设置画布，四个坐标系子图
    for picIndex in range(lenNumIter):  #遍历不同迭代次数
        for dataIndex in range(len(dataSet)): #遍历数据集
            classIndex=classMarkList[picIndex][dataIndex]-1 #该数据在本次聚类中的类别划分结果索引
            #绘制该样本点
            axs[picIndex].scatter(x=dataSet[dataIndex,0],y=dataSet[dataIndex,1],color=colors[classIndex],marker=markers[classIndex])
        axs[picIndex].set_xlabel('密度',fontproperties='SimHei')
        axs[picIndex].set_ylabel('含糖率',fontproperties='SimHei')
        axs[picIndex].set_title('迭代次数:%d'%numIterList[picIndex],fontproperties='SimHei')
    plt.show()
        
    
    
