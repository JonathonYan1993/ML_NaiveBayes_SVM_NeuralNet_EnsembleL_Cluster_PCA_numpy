# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 20:11:58 2019
本程序基于朴素贝叶斯分类器实现新闻分类
本程序朴素贝叶斯分类器通过调用sklearn.naive_bayes.MultinomialNB实现
参考资料：
https://github.com/Jack-Cherish/Machine-Learning/blob/master/Naive%20Bayes/nbc.py
@author: yanji
"""

from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import os
import random
import jieba

def textsPropcessing(folder_path,test_ratio=0.2):
    """
    function:中文文本处理，读取所有样本集的文本信息
    Parameters: folder_path - 文本存放的一级目录文件夹
               test_size - 测试集占比，默认设置占所有数据集的百分之20
    Returns: all_word_list -  按词频降序排序的训练集列表
             train_data_list - 训练集列表
             test_data_list - 测试集列表
             train_class_list - 训练集标签列表
             test_class_list - 测试集标签列表
    """
    folder_list = os.listdir(folder_path)  #读取一级目录文件夹下的所有二级文件夹列表
    data_list = [] #初始化数据集数据
    class_list = [] #初始化数据集标签类别
    #遍历每个子文件夹
    for folder in folder_list:
        new_folder_path=os.path.join(folder_path,folder) #将一级目录与二级目录文件夹合成新的路径
        files=os.listdir(new_folder_path)  #读取二级文件夹下的所有txt文件列表
        j=1 #初始化该二级文件夹（该类样本）的样本数量
        #遍历该二级文件夹下所有txt文件
        for file in files:
            #每类样本数量最多设置为100个
            if j>100:
                break
            with open(os.path.join(new_folder_path,file),'r',encoding='utf-8') as f: #打开txt文本文件
                fileWord=f.read()     #读取txt文本文件所有内容
            word_list=jieba.lcut(fileWord,cut_all=False) #精简模式分词，返回一个列表
            data_list.append(word_list)  #将数据添加到数据集里
            class_list.append(folder)   #将数据标签添加到数据集标签里
            j+=1
    data_class_list = list(zip(data_list,class_list)) #zip压缩合并，将数据与标签压缩合并
    random.shuffle(data_class_list)    #打乱data_class_list顺序
    indexSplit = int(len(data_class_list)*test_ratio)+1 #切分训练集与测试集的索引值
    train_list=data_class_list[indexSplit:]   #划分训练集
    test_list=data_class_list[:indexSplit]   #划分测试集
    train_data_list,train_class_list = zip(*train_list)  #训练集解压缩
    test_data_list,test_class_list = zip(*test_list) #测试集解压缩
    train_data_list=list(train_data_list)  #转化为list
    train_class_list=list(train_class_list)
    test_data_list = list(test_data_list)
    test_class_list = list(test_class_list)
    
    #统计训练集词频
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if  word in all_words_dict.keys():
                all_words_dict[word]+=1
            else:
                all_words_dict[word]=1
    #对词集按照键的数值倒序排序
    all_words_tuple_list=sorted(all_words_dict.items(),key=lambda f:f[1],reverse=True)
    all_words_list,all_words_nums=zip(*all_words_tuple_list)  #对词集字典进行解压缩
    all_words_list =list(all_words_list)  #转换成列表
    return all_words_list,train_data_list,train_class_list,test_data_list,test_class_list
    
def makeStopWordsSet(words_file):
    """
    function:读取停词表文件内容，去重
    parameters: words_file - 文件路径
    returns: words_set - 读取内容的set集合
    """
    words_set = set()  #创建不重复词集列表
    with open(words_file,'r',encoding='utf-8') as f:  #打开文件
        for line in f.readlines():   #一次性读取所有行，并逐行遍历
            line.strip() #去掉空格及换行符
            if len(line)>0: #长度大于0，说明有文本，则添加到词集中
                words_set.add(line) #对于停词表，一个词则为一行
    return words_set

def texts2Features(train_data_list,test_data_list,feature_words):
    """
    function:根据特征集feature_words将训练集与测试集向量化
    parameters: train_data_list - 训练集
                test_data_list - 测试集
                feature_words - 特征集
    Returns: train_feature_list - 训练集向量化列表
             test_feature_list - 测试集向量化列表
    """
    #定义一个函数，该函数创建一个feature_words尺寸大小的列表，如果特征词在该文本中，则该特征词相应位置标记为1，如此返回一个该文本对应的特征向量
    def text_feature(text,feature_words):
        text_words=set(text)  #对文本进行清洗，返回一个不重复的词汇列表
        feature=[1 if word in text_words else 0 for word in feature_words]
        return feature
    train_feature_list=[text_feature(text,feature_words) for text in train_data_list] #创建训练集向量列表
    test_feature_list=[text_feature(text,feature_words) for text in test_data_list] #创建测试集向量列表
    return train_feature_list,test_feature_list

def createFeatureWords(all_words_list,deleteN,stopwords_set=set()):
    """
    function:根据所有词汇列表，停词表，需要删除的高频词汇个数来创建特征词集表     
    Parameters: all_words_list - 根据所有txt文件创建的词频降序的全部词汇列表
                deleteN - 需要去掉的前deleteN个高频词汇（不作为特征词汇）
                stopwords_set - 指定的停词表
    Returns: feature_words - 特征词汇表
    """
    feature_words = [] #创建空的特征词汇表
    n=1 #初始化特征词汇个数
    for t in range(deleteN,len(all_words_list),1):
        if n>1000:
            break  #设定特征词汇表feature_words的最大维度为1000
        word=all_words_list[t]
        #如果这个词不是数字，也不是指定的停词结束语，且单词长度大于1小于5，则该词可作为特征词
        if not word.isdigit() and word not in stopwords_set and 1<len(word)<5:
            feature_words.append(word)
        n+=1
    return feature_words

def textNaiveBayesClassifier(train_feature_list,train_class_list,test_feature_list,test_class_list):
    """
    function:利用sklearn库创建朴素贝叶斯分类器，实现新闻分类
    Parameters: train_feature_list - 训练集向量化的特征文本
                train_class_list - 训练集分类标签
                test_feature_ilst - 测试集向量化的特征文本
                test_class_list - 测试集分类标签
    Returns:  test_accuracy - 分类器精度
    """
    #创建朴素贝叶斯分类器
    classifier=MultinomialNB().fit(train_feature_list,train_class_list)
    #基于测试集，计算该分类器的泛化性能，返回测试精度
    test_accuracy = classifier.score(test_feature_list,test_class_list)
    return test_accuracy

if __name__=='__main__':
    #文本预处理
    folder_path='news/Sample' #所有数据存放地址
    #读取数据集的所有文本信息
    all_words_list,train_data_list,train_class_list,test_data_list,test_class_list=textsPropcessing(folder_path,test_ratio=0.2)
    #读取并生成停词表
    stopwords_file='news/stopwords_cn.txt'
    stopwords_set = makeStopWordsSet(stopwords_file)
    
    test_accuracy_list=[]
    deleteNs=range(0,1000,20) #创建一个高频词汇去除个数列表
    for deleteN in deleteNs:
        #创建特征词汇表
        feature_words=createFeatureWords(all_words_list,deleteN,stopwords_set)
        #将训练集与测试集向量化
        train_feature_list,test_feature_list= texts2Features(train_data_list,test_data_list,feature_words)
        #基于训练集创建分类器，计算测试集分类精度
        test_accuracy = textNaiveBayesClassifier(train_feature_list,train_class_list,test_feature_list,test_class_list)
        test_accuracy_list.append(test_accuracy)
    plt.figure()
    #绘制deleteN与test_accuracy的关系曲线
    plt.plot(deleteNs,test_accuracy_list,Color='blue')
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()    
        
    


    
        
    
    
    
            
        
        
    
    
