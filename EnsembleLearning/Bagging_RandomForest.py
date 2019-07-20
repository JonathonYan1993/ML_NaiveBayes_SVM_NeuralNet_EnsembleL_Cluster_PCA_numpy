# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:08:37 2019
本程序采用集成学习方法对数据集进行划分
所采用方法为Bagging与随机森林Random Forest，改变集成规模，观察两种方法泛化误差的变化
Bagging也采用决策树为基学习器，其中Bagging使用“确定型”决策树，随机森林使用“随机型”决策树
数据集为sonar声呐数据集
参考资料：周志华《机器学习》第八章8.1、8.3、8.4节
@author: yanji
"""

import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    """
    function:读取数据集，本程序读取sonar声纳数据集
    Parameters:fileName - 存储数据集的文件名
    Returns: dfData - 读取的数据集，以列表的形式返回
    """
    dfData=pd.read_csv(fileName)  #读取数据集
    listData=dfData.values.tolist() #将dataFrame结构数据转化为列表
    return listData 

def dataSampling(dataSet,numTrain,scaleEL):
    """
    function:从数据集中随机抽样抽取一定数量的数据不相同的训练集，未被抽中的数据组成测试集
    Parameters: dataSet - 原始数据集
                numTrain - 训练集数据个数
                scaleEL - 训练集数目，即集成学习集成规模
    Returns: ELData - 用于集成学习的数据
             #testData - 用于测试的数据
    """
    #lenData=len(dataSet)  #原始数据集长度
    #signData=np.zeros(lenData,np.int)  #原始数据集标识位，初始化为0，表示未被抽到训练集中，抽到后置为1
    ELData=[] #初始化集成学习训练集
    for numIter in range(scaleEL):
        nowData=[] #初始化本次抽取的训练集
        for i in range(numTrain):
            ranInd = round(random.uniform(0,160)) #随机产生一个0～lenData之间的索引值,将最大索引改为180，后面剩余数据都将作为测试数据
            nowData.append(dataSet[ranInd])  #扩展本次训练集数据
            #signData[ranInd]=1  #将该索引位置的标识位值1
        ELData.append(nowData)  #扩展集成学习训练集
    #testData=dataSet[160:]  #初始化测试集
    """
    for i in range(lenData):
        if signData[i]==0:    #如果标识位为0，则该数据未被抽中，将其添加到测试集中
            testData.append(dataSet[i])  #扩展测试集
    """
    return ELData#,testData

def giniCalc(groups,values):
    """
    function: 计算数据集分成groups后的基尼系数
    Parameters: groups - 依据某属性某阈值划分后的数据分组
                values - 数据的类别种类
    Returns: gini - 数据集划分后的基尼系数
    """
    gini=0.0 #初始化基尼系数
    sizeSum=0 #初始化数据集总的数据个数
    for group in groups: #遍历数据分组
        sizeGroup=len(group) #该组数据个数
        if sizeGroup ==0:  #如果该组数据个数为0，跳过该组
            continue
        sizeSum+=sizeGroup   #更新总数据个数
        groupValue=[row[-1] for row in group] #提取该组数据的类别
        valueProportion=[groupValue.count(value)/sizeGroup for value in values] #计算各类别在该组数据中的比例
        giniGroup=1-sum([pow(p,2) for p in valueProportion])#计算该组数据的基尼值
        gini+=sizeGroup*giniGroup #更新基尼系数
    return gini/sizeSum #返回基尼系数

def dataSplit(index,value,dataSet):
    """
    function:依据属性索引，以所设定属性取值对数据集实现二分类
    Parameters: index - 划分属性的索引
                value - 划分属性的划分阈值
                dataSet - 待划分数据集
    Returns: left - 左分类（决策树形象）数据集
             right - 右分类数据集
    """
    left=list() #初始化左分类数据集
    right=list() #初始化右分类数据集
    for row in dataSet: #遍历待划分数据集
        if row[index]<value: #该属性取值小于阈值，划分到左数据集，否则划分到右数据集
            left.append(row)
        else:
            right.append(row)
    return left,right #返回划分结果

def dataSet2Split(dataSet):
    """
    function: 对数据集选择最佳划分属性及划分阈值，实现数据集二分类
    Parameters: dataset - 待划分数据集
    Returns: b_index - 最佳分类属性索引
             b_value - 划分属性阈值
             b_groups - 划分结果
    """
    classValues=list(set([row[-1] for row in dataSet])) #统计数据集类别标签
    b_index,b_value,b_score,b_groups=999,999,999,None  #初始化划分属性，划分属性取值，划分结果基尼系数，划分结果
    if ELMethod=='R':  #如果ELMethod=R，则集成方法为随机森林，则在目前数据集中随机选择log2d个属性，作为属性集，最优划分属性在该属性集里选取
        indexRange=[]  #初始化属性索引列表
        indexLen=len(dataSet[0])-1  #数据集的整体属性个数
        indexRangeLen=int(math.log2(indexLen)) #随机选取的属性个数
        for i in range(indexRangeLen):  
            indexNow=int(random.uniform(0,indexLen-1))  #随机选择一个属性索引，如果已经在随机属性集里则重新选取
            while (indexNow in indexRange):
                indexNow=int(random.uniform(0,indexLen-1))
            indexRange.append(indexNow)  #将选择的属性添加到随机属性集里
    else:
        indexRange=list(range(len(dataSet[0])-1))  #如果不是随机森林，则应该为Bagging或者其他方法，此时属性集为所有属性
    for index in indexRange:  #遍历每个属性
        for row in dataSet:   #遍历同属性下每个取值
           groups=list(dataSplit(index,row[index],dataSet)) #依据属性，及属性取值对数据集进行二分类
           gini=giniCalc(groups,classValues) #计算划分结果的基尼系数
           if gini<b_score:  #选择基尼系数最小的划分属性及划分属性取值
               b_index=index
               b_value=row[index]
               b_score=gini
               b_groups=groups
    #print({'index':b_index,'value':b_value})
    return {'index':b_index,'value':b_value,'groups':b_groups} #返回划分属性，划分属性取值，划分结果
    
def classChose(group):
    """
    function:选择一组数据中数量最多的类别，作为该组数据的类别划分结果
    Parameters: group - 待确定类别标签的数据组
    Returns: classType - 该组数据的类别
    """
    outcomes = [row[-1] for row in group] #统计该组数据各数据的类别标签
    classType = max(set(outcomes),key=lambda x:outcomes.count(x)) #选择数量最多的类别标签
    return classType

def treeGenerate(node,max_depth,min_size,depth):
    """
    function:基于上一个根节点上的groups，继续创建根节点或叶节点递归循环以完善决策树
    Parameters: node - 决策树上一个根节点
                max_depth - 决策树最大层数阈值，当决策树层数达到阈值时，停止划分，直接确定剩余数据组类别，完善决策树
                min_size - 数据组最小数据个数，当数据个数小于等于该阈值时，不再划分，直接确定类别标签
                depth - 决策树上一个节点的所在层数
    Returns:  无返回
    """
    left,right=node['groups'] #提取上一个根节点划分的左右数据组
    del(node['groups'])  #删除决策树字典中不再有用的数据组
    if not left or not right: #如果左数据组或右数据组为空，则左右叶节点返回相同的类别标签
        node['left']=node['right']=classChose(left+right)
        return
    if depth>=max_depth: #如果决策树层数达到最大值时，不再继续划分，直接确定左右数据组的类别
        node['left']=classChose(left)
        node['right']=classChose(right)
        return
    if len(left)<=min_size:
        node['left']=classChose(left)  #如果左数据组数据个数低于下限，不再划分，直接返回改组类别
    else:
        node['left']=dataSet2Split(left) #对左数据组继续划分
        treeGenerate(node['left'],max_depth,min_size,depth+1) #递归调用，依据划分结果继续完善决策树
    if len(right)<=min_size:
        node['right']=classChose(right)  #如果右数据组数据个数低于下限，不再划分，直接返回该组类别
    else:
        node['right']=dataSet2Split(right) #对右数据组继续划分
        treeGenerate(node['right'],max_depth,min_size,depth+1) #递归调用，依据划分结果继续完善决策树
                
def treeBuild(dataSet,max_depth,min_size):
    """
    function: 依据数据集创建决策树
    Parameters: dataSet - 用于创建决策树的数据集
                max_depth - 决策数的最大层数
                min_size - 数据组继续划分的数据个数下限，但数据个数小于等于该值时，数据不再划分
    Returns: treeGini - 创建的决策树
    """
    treeGini=dataSet2Split(dataSet) #创建第一层根节点
    treeGenerate(treeGini,max_depth,min_size,1) #基于第一个根节点划分的数据继续划分，完善决策树
    return treeGini  #返回决策树

def classPredict(node,dataNew):
    """
    function: 依据已经建立的决策树，预测新数据的类别
    Parameters: node - 已建立的决策树后者决策树的某个子树
                dataNew - 新的数据样本
    Returns: classType - 该数据类别标签预测结果
    """
    if dataNew[node['index']]<node['value']: #判断划分属性取值是否小于阈值，是则取left的结果，否则取right的结果
        if isinstance(node['left'],dict):  #判断node[left]是否是字典，否则返回类别，是则递归循环调用进行匹配
            classType=classPredict(node['left'],dataNew)
        else:
            classType=node['left']
    else:
        if isinstance(node['right'],dict):  #判断node[right]是否是字典，否则返回类别，是则递归循环调用进行匹配
            classType=classPredict(node['right'],dataNew)
        else:
            classType=node['right']
    return classType #返回预测结果            
        
def treeELBuild(dataSetEL,scaleEL,max_depth,min_size):
    """
    function:集成学习，基于集成数据集与集成规模，创建多个决策树
    Parameters: detaSetEL - 集成数据集
                scaleEL - 集成规模
                max_depth - 决策数的最大层数
                min_size - 数据组继续划分的数据个数下限，但数据个数小于等于该值时，数据不再划分
    Returns: treeEL - 集成学习生成的决策树数列
    """
    treeEL=[] #初始化决策树数列
    for i in range(scaleEL): #遍历集成数据集，依据每个数据集创建一个决策树，添加到数列中
        treeEL.append(treeBuild(dataSetEL[i],max_depth,min_size))
    return treeEL #返回决策树数列

def treeELAccuCalc(treeEL,scaleEL,testData):
    """
    function: 利用集成学习生成的决策树数列对测试集进行预测，并采用投票法进行最终预测，计算测试集的预测准确率
    Parameters: treeEL - 集成学习生成的决策树数列
                scaleEL - 集成规模
                testData - 测试集
    Returns: errorEL - 测试集的预测错误率
    """
    errorEL=0 #初始化预测错误个数
    for row in testData:  #遍历测试集每个数据
        classPre=[] #初始化各个学习器的预测结果
        for i in range(scaleEL): #遍历各个学习器，将每个学习器对该样本的预测结果添加到预测列表中
            classPre.append(classPredict(treeEL[i],row))
        classData=max(set(classPre),key=lambda x: classPre.count(x)) #从预测列表中找出数目最多的预测类别作为最终预测结果
        if classData != row[-1]:  #如果预测结果与真实结果不一致，打印输出
            #print('error: Real:',row[-1],'Predict:',classData)
            errorEL+=1 #预测错误个数加1
    return errorEL/len(testData) #返回预测错误率

if __name__=='__main__':
    #读取声呐数据集
    sonarData=loadDataSet("sonar.all-data.csv")
    #打乱原始数据集
    random.shuffle(sonarData)
    numTrain = 120 #训练集数据个数
    #scaleEL=5 #集成规模
    testData=sonarData[160:]  #测试集
    max_depth=5 #决策树最大层数
    min_size =10 #数据进一步划分的最小规模
    ELMethod='B' #初始化集成方法标识
    #改变集成规模，观察泛化误差随集成规模的变化情况
    errorBagging=[] #初始化Bagging集成学习误差数列
    errorRF=[] #初始化随机森林集成学习误差数列
    for scaleEL in range(1,50,1):
        #对原始数据进行随机采样，得到总的集成数据集
        DataEL=dataSampling(sonarData,numTrain,scaleEL)
        #创建Bagging集成学习决策树数列
        ELMethod='B'
        treeBagging=treeELBuild(DataEL,scaleEL,max_depth,min_size)
        #计算Bagging集成学习的泛化误差
        errorBagging.append(treeELAccuCalc(treeBagging,scaleEL,testData))
        #创建随机森林集成学习决策树数列
        ELMethod='R'
        treeRF=treeELBuild(DataEL,scaleEL,max_depth,min_size)
        #计算Bagging集成学习的泛化误差
        errorRF.append(treeELAccuCalc(treeRF,scaleEL,testData))
    plt.plot(list(range(1,50,1)),errorBagging,color='red',label='Bagging') #绘制Bagging误差曲线
    plt.plot(list(range(1,50,1)),errorRF,color='c',label='Random Forest') #绘制随机森林误差曲线
    plt.legend() #添加图例
    plt.xlabel('scale')  #x轴标签
    plt.ylabel('error(%)') #y轴标签
    plt.title('Influence by EL scale') #图题
    plt.show() #显示
        

