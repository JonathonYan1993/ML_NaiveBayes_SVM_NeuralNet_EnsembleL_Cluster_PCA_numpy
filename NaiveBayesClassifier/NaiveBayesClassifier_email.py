# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 15:29:28 2019
本程序基于朴素贝叶斯分类器实现垃圾邮件的分类
参考资料：
周志华《机器学习》第7章
https://github.com/Jack-Cherish/Machine-Learning/tree/master/Naive%20Bayes
@author: yanji
"""

import numpy as np
import random
import re

def loadDataSet():
    """
    function:创建实验样本
    Parameters: 无
    Returns: postingList - 实验样本切分的词条
             classVec - 类别标签向量
    """
    #切分的词条
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1] #词条样本分类
    return postingList,classVec

def createVocabList(dataSet):
    """
    function:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
    Parameters: dataSet - 整理的样本数据集
    Returns: vocabSet - 返回不重复的词条列表，也就是词汇表
    """
    vocabSet=set([]) #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document) # 取不重复的词条并集
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    """
    function:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0
    Parameters: vocabList - createVocabList返回的列表
                inputSet - 切分的词条列表
    Returns: returnVec - 文档向量，词集模型
    """
    #根据词汇表创建一个其中所有元素都为0的文档向量
    returnVec=[0]*len(vocabList)
    #遍历每个词条
    for word in inputSet:
        if word in vocabList:
            #如果词条存在于词汇表中，则该词条对应的文档向量相应元素位置置1
            returnVec[vocabList.index(word)]=1
        else:
            print("the word: %s isn't in my Vocabulary!"%word)
    return returnVec  #返回文档向量

def trainNaiveBayes(trainMatrix,trainCategory):
    """
    function:朴素贝叶斯分类器训练函数
    Parameters: trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec矩阵
                trainCategory - 训练类别标签，即loadDataSet返回的classVec
    Returns: p0Vec - 非侮辱类的条件概率数组
             p1Vec - 侮辱类的条件概率数组
             pAbusive - 文档属于侮辱类的概率
    """
    numTrainDocs = len(trainMatrix)  #计算训练的文档数目(样本个数)
    numWords = len(trainMatrix[0])  #计算每篇文档的词条数(属性个数)
    pAbsuive = (sum(trainCategory)+1)/(float(numTrainDocs)+2) #文档属于侮辱类的概率，拉普拉斯修正，种类数为2
    p0Num = np.ones(numWords) #初始化非侮辱类的条件概率数组，所有属性取值为1，拉普拉斯修正，初始化为1
    p1Num = np.ones(numWords) #初始化侮辱类的条件概率数组，所有属性取值为1，拉普拉斯修正，初始化为1
    p0Denom=float(numTrainDocs)-sum(trainCategory)+2   #分母初始化,拉普拉斯修正
    p1Denom=sum(trainCategory)+2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
        else:
            p0Num += trainMatrix[i]
    p1Vec = p1Num/p1Denom  #类别为1，各属性取值为1的类条件概率向量
    p0Vec = p0Num/p0Denom  #类别为0，各属性取值为0的类条件概率向量
    return p0Vec,p1Vec,pAbsuive

def classifyNaiveBayes(vec2Classify,p0Vec,p1Vec,pClass1):
    """
    function:朴素贝叶斯分类器分类函数
    Parameters: vec2Classify - 待分类的词条数组
                p0Vec - 非侮辱类的条件概率数组，所有属性取值为1
                p1Vec - 侮辱类的条件概率数组，所有属性取值为1
                pClass1 - 文档属于侮辱类的概率
    Returns: 0 - 属于非侮辱类
             1 - 属于侮辱类
    """
    p1=pClass1 #初始化该测试例为侮辱类的概率
    p0=1-pClass1 #初始化该测试例为非侮辱类的概率
    numProp=len(vec2Classify)
    for i in range(numProp):
        #如果属性取值为1，则直接乘以对应的类条件概率，否则类条件概率为1-pVec[i]
        if vec2Classify[i]==1:
            p1=p1*p1Vec[i]
            p0=p0*p0Vec[i]
        else:
            p1=p1*(1-p1Vec[i])
            p0=p0*(1-p0Vec[i])
    print('p1:',p1)
    print('p0:',p0)
    #如果属于侮辱类的概率大，则返回侮辱类（垃圾）1，否则返回非侮辱类（非垃圾）0
    if p1>p0:
        return 1
    else:
        return 0
    
def testingNaiveBayes():
    """
    function:利用loadDataSet的实验样本测试朴素贝叶斯分类器
    Parameters:
    Returns:
    """
    listOPosts,listClasses=loadDataSet()  #获得实验样本及其分类
    myVcoabList=createVocabList(listOPosts) #创建词汇表
    trainMat=[] #初始化训练样本文档向量
    for listPost in listOPosts:
        trainMat.append(setOfWords2Vec(myVcoabList,listPost)) #获得训练样本文档向量
    p0V,p1V,pAb =  trainNaiveBayes(trainMat,listClasses) #训练朴素贝叶斯分类器
    
    testEntry = ['love', 'my', 'dalmation']  #利用非侮辱类测试例测试分类器	
    thisDoc = setOfWords2Vec(myVcoabList,testEntry)
    if classifyNaiveBayes(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')
        
    testEntry = ['stupid', 'garbage']	  #利用侮辱类测试例测试分类器
    thisDoc = setOfWords2Vec(myVcoabList,testEntry)
    if classifyNaiveBayes(thisDoc,p0V,p1V,pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry,'属于非侮辱类')
        
def textSplit(bigString):
    """
    function:接收一个大字符串，并将其解析为字符串列表
    Parameters: bigString - 大字符串
    Returns: 字符串列表
    """
    listOfTokens =  re.split(r'\W*',bigString) #将大字符串以非字母字符为切分标志，对字符串进行切分，得到单词列表
    return [tok.lower() for tok in listOfTokens if len(tok)>2] #将长度大于2的单词变成小写
        
def emailNaiveBayesClassify():
    """
    function:利用朴素贝叶斯分类器对垃圾邮件进行分类，并进行测试
    Parameters:
    Returns:
    """
    docList=[]  #储存文档字符串列表
    classList=[] #储存文档类型，1表示垃圾邮件，0表示非垃圾邮件
    for i in range(1,26):
        wordList=textSplit(open('email/spam/%d.txt'%i,'r').read()) #读取垃圾邮件文件夹所有垃圾邮件内容，转化为字符串列表
        docList.append(wordList) #将字符串列表添加到文档字符串列表
        classList.append(1) #该垃圾邮件类别标记为1
        wordList=textSplit(open('email/ham/%d.txt'%i,'r').read()) #读取非垃圾邮件文件夹所有非垃圾邮件内容，转化为字符串列表
        docList.append(wordList) #将字符串列表添加到文档字符串列表
        classList.append(0) #该非垃圾邮件类别标记为0
    vocabList=createVocabList(docList) #基于文档字符串列表创建词汇表
    trainSetIndex=list(range(50)) #初始化训练集样本索引列表
    testSetIndex=[] #初始化测试集样本索引列表
    #随机抽取10个索引值，作为测试集样本索引值
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainSetIndex)))
        testSetIndex.append(trainSetIndex[randIndex]) #添加测试集索引值
        del(trainSetIndex[randIndex]) #从训练集索引列表中删除该索引值，保证了测试集样本不会与自身及训练集重复
    trainMat=[] #初始化训练集
    trainClasses=[] #初始化训练集类别
    for docIndex in trainSetIndex:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex])) #依据训练集索引值，转化该样本文档向量，添加到训练集中
        trainClasses.append(classList[docIndex]) #添加训练集元素对应类别标记
    p0V,p1V,pAb =  trainNaiveBayes(trainMat,trainClasses) #训练朴素贝叶斯分类器
    
    errorCount=0 #初始化测试集中分类错误个数
    for docIndex in testSetIndex:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])#依次将测试集样本转化为文档向量
        typeClass= classifyNaiveBayes(wordVector,p0V,p1V,pAb) #对测视例进行分类
        print(docIndex,classList[docIndex],typeClass) #输出文档索引，真实类别，预测类别
        if typeClass != classList[docIndex]:
            errorCount+=1 #累积识别错误个数
    print('错误率：%.2f%%'%(float(errorCount)/len(testSetIndex)*100)) #输出测试集识别错误率
        
if __name__ =='__main__':
    #testingNaiveBayes()
    emailNaiveBayesClassify()
    
    
        
    

    

    
