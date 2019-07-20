# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 17:33:02 2019
本程序基于软间隔支持向量机实现数据分类，优化算法为完整的Platt SMO算法
数据集为testSet.txt
参考资料：
周志华《机器学习》第六章6.1～6.4节
李航《统计学习方法》第七章7.4节
Peter Harrington《机器学习实战》第六章6.3、6.4节
博客：https://cuijiahua.com/blog/2017/11/ml_8_svm_1.html机器学习实战教程（八）：支持向量机原理篇之手撕线性SVM
博客：https://www.cnblogs.com/pinard/p/6111471.html支持向量机原理(四)SMO算法原理 
https://github.com/Jack-Cherish/Machine-Learning/blob/master/SVM/svm-smo.py
@author: yanji
"""

import matplotlib.pyplot as plt
import numpy as np
import random

class optStruct:
    """
    function:声明一个类，该类储存并维护数据、类别等所有需要操作的值
    Parameters: dataMatIn - 数据矩阵X
                classLabels - 数据标签y
                C - 惩罚系数
                toler - 容错率，软间隔的范围或者松弛变量kexi的临界值
    """
    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X = dataMatIn  #数据矩阵
        self.labelMat = classLabels  #数据类别标签
        self.C = C #惩罚系数
        self.tol = toler  #容错率
        self.m = np.shape(dataMatIn)[0] #数据矩阵的行数，即样本个数
        self.alphas = np.mat(np.zeros((self.m,1)))  #根据数据样本个数创建alpha参数矩阵，初始化为0
        self.b = 0 #位移项b初始化为0
        self.eCache = np.mat(np.zeros((self.m,2))) #根据数据样本个数创建一个便于寻找alpha_j的误差缓存矩阵，第一列为0/1是否有效的标志位，第二列为实际的误差值E

def loadDataSet(fileName):
    """
    function: 根据文件名读取相应文件的数据
    Parameters: fileName - 文件名
    Returns: dataMat - 数据矩阵
             labelMat - 数据类别标签
    """
    dataMat=[] #创建空的数据矩阵
    labelMat=[] #创建空的数据类别标签
    fr=open(fileName) #读取文件
    for line in fr.readlines():  #读取所有行，并逐行遍历处理数据
        lineArr = line.strip().split() #去掉前后的空格及换行符，以空格或制表符进行字符分割
        dataMat.append([float(lineArr[0]),float(lineArr[1])]) #将前两个数据添加到数据矩阵中，作为一个样本的属性取值
        labelMat.append(float(lineArr[2])) #将第三个数据，即类别标签添加到类别标签矩阵里
    return dataMat,labelMat

def calcEk(oS,k):
    """
    function:依据现有的alphas计算对第k个样本预测，计算真实结果与预测结果之间的误差
    Parameters: oS - 现有的数据类
                k -  待求误差的数据索引k
    Returns: Ek - 索引为k的数据误差
    """
    fxk = float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)+oS.b)
    Ek=fxk - float(oS.labelMat[k])
    return Ek

def selectJrand(i,m):
    """
    function: 随机选择alpha_j的索引值，该函数应用于第一次寻找alpha_j时
    Parameters: i - alpha_i的索引值
                m - alphas的参数个数，即样本个数
    Returns: j - alpha_j的索引值
    """
    j=i #初始化j为i
    while(j==i):
        #在0～m间寻找一个不等于i的随机数j
        j=int(random.uniform(0,m))
    return j

def selectJ(i,oS,Ei):
    """
    function: 采用内循环启发方式2选取alpha_j的索引值，即选取使|Ei-Ej|最大的alpha_j
    Parameters: i - 第i个数据的索引值
                oS - 现有的数据类
                Ei - 第i个数据目前的预测误差值
    Returns: j,maxK - 寻找到的alpha_j的索引值
             Ej - 第j个数据目前的预测误差值
    """
    maxK=-1 #初始化最大误差间隔的数据索引值
    maxDeltaE=0 #初始化最大误差间隔
    Ej=0 #初始化alpha_j的误差
    oS.eCache[i]=[1,Ei] #将Ei存入误差缓存矩阵，或者对误差缓存矩阵Ei进行更新
    #求取误差不为0（或者说存在误差）的数据的索引值，将误差矩阵转为数列array，求取第一列标志数据不为零的索引，由于只有一维，则求取结果第一个值即为所求
    validEcacheList = np.nonzero(oS.eCache[:,0].A)[0] 
    if(len(validEcacheList)>1): #判断长度是否大于1，即除了第i位Ei是否还有其他的误差不为0的数据
        #如果大于1，则存在其他不为0的误差，寻找最大误差间隔
        for k in validEcacheList:  #遍历寻找最大误差间隔
            if k == i:    #跳过i
                continue
            Ek=calcEk(oS,k) #计算目前alphas下第k个数据的预测误差Ek
            deltaE=abs(Ei-Ek) #计算误差间隔|Ei-Ek|
            if(deltaE>maxDeltaE):
                maxK =k
                maxDeltaE = deltaE
                Ej=Ek
        return maxK,Ej #返回得到最大误差间隔的数据索引值作为alpha_j的索引以及误差Ej
    else:
        #如果不大于1，说明目前只有第i位Ei非0，此时采用随机索引方法选择alpha_j的索引值
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j) #计算Ej
    return j,Ej #返回索引j与误差Ej

def updateEk(oS,k):
    """
    function: 计算第k个样本数据的预测误差，并更新误差缓存数据
    Parameters: oS - 现有的数据类
                k - 第k个样本数据的索引值
    Returns: 无
    """
    Ek=calcEk(oS,k)   #计算目前alphas下第k个数据的预测误差Ek
    oS.eCache[k]=[1,Ek] #更新误差缓存矩阵第k行数据
    
def clipAlpha(aj,H,L):
    """
    function: 对alpha_j_new_unc进行修剪，得到alpha_j_new
    Parameters: aj - 待修剪的alpha_j_new_unc值
                H - alpha上限
                L - alpha下限
    Returns: aj - 修剪后的alpha_j_new值
    """
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj

def innerL(i,oS):
    """
    function:优化的SMO算法，对取定的alpha_i，若能优化，找到alpha_j，进行双变量优化
    Parameters: i - 第i个数据的索引值
                oS - 现有的数据类
    Returns: 1 - 有任意一对alpha值发生变化，即进行了一次优化
             0 - 没有任意一对alpha值发生变化，或者变化量太小，即未进行优化
    """
    #步骤1：计算误差Ei
    Ei=calcEk(oS,i)
    #优化alpha,设定一定的容错率
    """
    以下为本人对是否违反KKT条件判断语句的理解：
    设定预测值ui=w*xi+b
    KKT条件为：1、alphai=0时,yiui>=1 2、0<alphai<C时,yiui=1 3、alphai=C时,yiui<=1
    由于yiyi=1，则KKT条件可以改写为：1、alphai=0时，yi(ui-yi)=yiEi>=0 2、0<alphai<C时,yiEi=0 3、alphai=C时,yiui<=0
    则违反KKT条件的情形为：
    当alphai<C时，yiEi<0,即本该在间隔边界上或间隔边界远离超平面一侧，现在到间隔边界靠近超平面一侧了
    当alphai>0时，yiEi>0,即本该在间隔边界上或间隔边界靠近超平面一侧，现在到间隔边界原理超平面一侧了
    硬间隔则可直接判断违反了KKT条件，软间隔则可以设定一个允许跨越的阈值，即容错率，从另一角度也可以理解为松弛变量kexi的临界值
    此时违反KKT条件的情形为：
    当alphai<C时，yiEi<-toler
    当alphai>0时，yiEi>toler
    """
    if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei>oS.tol) and (oS.alphas[i]>0)):
        #使用内循环启发方式2选择alpha_j，并计算Ej
        j,Ej=selectJ(i,oS,Ei)
        #使用深拷贝，保存更新前的alpha值
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        #步骤2：计算上界H和下界L
        """
        上界与下界的理解：
        以yiyj异号为例，alphajnew-alphainew=alphajold-alphaiold
        则alphajnew=alphainew+alphajold-alphaiold
        由于0<=alpha<=C,可以得到两个约束:
        0<=alphajnew<=C
        alphajold-alphaiold<=alphajnew=alphainew+alphajold-alphaiold<=C+alphajold-alphaiold
        因此有下面的L,H求解，yiyj同号时同理
        """
        if(oS.labelMat[i]!=oS.labelMat[j]):
            L=max(0,oS.alphas[j]-oS.alphas[i])
            H=min(oS.C,oS.C+oS.alphas[j]-oS.alphas[i])
        else:
            L=max(0,oS.alphas[j]+oS.alphas[i]-oS.C)
            H=min(oS.C,oS.alphas[j]+oS.alphas[i])
        if L==H:
            #L=H，则alphaj,alphai一个为0，一个为C；或者同时为0；或者同时为C，继续优化依然是原值，没有必要
            print("L=H") 
            return 0
        #步骤3:计算eta
        eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        if eta>=0:
            print("eta>=0")
            return 0
        #步骤4:更新alpha_j
        oS.alphas[j]-=oS.labelMat[j]*(Ei-Ej)/eta
        #步骤5:修剪alpha_j
        oS.alphas[j]=clipAlpha(oS.alphas[j],H,L)
        #更新Ej至误差缓存
        updateEk(oS,j)
        #如果alpha_j变化太小，则不对alpha_i更新，即不进行一对变量的优化
        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            print("alpha_j变化太小")
            return 0
        #步骤6：更新alpha_i
        oS.alphas[i]+=oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        #更新Ei至误差缓存
        updateEk(oS,i)
        #步骤7：更新b_1和b_2
        b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        #步骤8:依据b_1和b_2更新b，即选择在间隔边界上的点计算b，若都不在间隔边界上，取平均值
        if(0<oS.alphas[i])and(oS.alphas[i]<oS.C):
            oS.b=b1
        elif(0<oS.alphas[j])and(oS.alphas[j]<oS.C):
            oS.b=b2
        else:
            oS.b=(b1+b2)/2.0
        return 1
    else:
        return 0
    
def smoP(dataMatIn,classLabels,C,toler,maxIter):
    """
    function: 完整的线性SMO算法，根据设定的参数对现有的数据进行划分
    Parameters:  dataMatIn - 数据矩阵
                 classLabels - 数据类别标签
                 C - 惩罚系数
                 toler - 容错率，软间隔的范围或者松弛变量kexi的临界值
                 maxIter - 最大alphas遍历次数
    Returns: oS.b - SMO算法计算的位移项b
             oS.alphas - SMO算法计算的参数矩阵alphas
    """
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler) #依据现有数据和参数创建数据类
    iter = 0 #初始化遍历次数为0
    entireSet = True #初始化是否遍历整个样本集，为True则遍历整个样本集，为False则遍历非边界alpha值
    alphaPairsChanged=0 #初始化优化次数，每次遍历前都会置0，遍历后依然为0则未发生优化，大于0则该次遍历产生了优化
    #当遍历次数超过设定的最大值或者在遍历整个样本集后都未对任意alpha进行修改时，则退出循环
    while(iter<maxIter)and((alphaPairsChanged>0)or(entireSet)):
        alphaPairsChanged = 0 #优化次数置0
        if entireSet:
            #为True则遍历整个样本集
            for i in range(oS.m):
                alphaPairsChanged +=innerL(i,oS) #使用优化的SMO算法对alpha_i进行优化
                print("全样本遍历:第%d次遍历 样本%d,alpha优化次数:%d" %(iter,i,alphaPairsChanged))
            iter+=1 #遍历次数加1
        else:
            #为False则遍历非边界alpha值
            #寻找非边界alpha值，True计算时为1，False计算时为0，因为只有一维，取nonzero计算结果第一个值则为索引序列
            nonBoundIs = np.nonzero((oS.alphas.A>0)*(oS.alphas.A<C))[0]
            for i in nonBoundIs:
                alphaPairsChanged +=innerL(i,oS) #使用优化的SMO算法对alpha_i进行优化
                print("非边界遍历:第%d次遍历 样本%d,alpha优化次数:%d" %(iter,i,alphaPairsChanged))
            iter+=1
        """
        当entireSet为True时，即遍历了一次整个样本集，将其置为False，
        则在外部循环判断时，若alphasPairChanged>0,说明进行了优化，下次循环遍历非边界alpha值，若alphasPairChanged=0，则说明未进行优化，可退出外部循环
        当entireSet为False时，即遍历了一次非边界alpha值，
        如果alphasPairChanged>0时，说明进行了优化，下次外部循环依然遍历非边界alpha值
        如果alphasPairChanged=0时，说明未进行优化，将entireSet置为True，下次外部循环遍历整个样本集
        """
        if entireSet:
            entireSet = False
        elif(alphaPairsChanged==0):
            entireSet = True
        print("遍历次数:%d" %iter)
    return oS.b,oS.alphas   #返回SMO算法计算的位移项b和参数矩阵alphas

def calcWs(alphas,dataArr,classLabels):
    """
    function:依据参数矩阵alphas,数据矩阵X，数据类别标签y计算模型系数w
    Parameters: alphas - 参数矩阵alphas值
                dataArr - 数据矩阵X
                classLabels - 数据类别标签y
    Returns: w - 计算得到的模型系数
    """
    X=np.mat(dataArr) #将数据转为数据矩阵X
    labelMat=np.mat(classLabels).transpose() #将类别标签转为矩阵
    m,n=np.shape(X) #获取数据矩阵的大小，一维数目m为数据个数即样本个数，二维数目n为属性个数，即为模型系数w的维度
    w=np.zeros((n,1)) #初始化模型系数w为0
    for i in range(m):
        #依据样本数据与参数逐个累积计算w
        w += np.multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w  #返回模型系数w
    
def showClassifer(dataMat,classLabels,w,b):
    """
    function:将分类结果可视化，包括各个数据点，支持向量点以及划分超平面
    Parameters: dataMat - 数据矩阵X
                classLabels - 数据类别标签y
                w - 划分超平面模型系数
                b - 划分超平面位移项
    Returns: 无
    """
    #绘制各个数据样本点
    data_plus=[] #初始化正样本点
    data_minus=[] #初始化负样本点
    #找出各个正负样本点
    for i in range(len(dataMat)):
        if classLabels[i]>0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus) #将正样本数据转化为矩阵
    data_minus_np = np.array(data_minus) #将负样本数据转化为矩阵
    plt.scatter(np.transpose(data_plus_np)[0],np.transpose(data_plus_np)[1],s=30,alpha=0.7) #绘制正样本散点图
    plt.scatter(np.transpose(data_minus_np)[0],np.transpose(data_minus_np)[1],s=30,alpha=0.7) #绘制负样本点散点图
    #绘制划分超平面直线
    x1=max(dataMat)[0] #寻找直线上右端点x（即x最大)
    x2=min(dataMat)[0] #寻找直线上左端点x（即x最小）
    a1,a2=w #将模型系数的两个值分开，a1对应第一个属性值，即坐标系上x;a2对应第二个属性值，即坐标系上y
    b=float(b) 
    a1=float(a1[0]) #将1X1矩阵的值提取出来
    a2=float(a2[0])
    #根据直线方程Ax+By+C=0计算x1,x2对应的y1,y2
    y1=(-b-a1*x1)/a2
    y2=(-b-a1*x2)/a2
    plt.plot([x1,x2],[y1,y2])
    #绘制支持向量点
    for i,alpha in enumerate(alphas):
        if alpha >0: #当参数alpha>0时，该样本点为支持向量点
            x,y=dataMat[i]
            plt.scatter([x],[y],s=150,c='none',alpha=0.7,linewidth=1.5,edgecolor='red')
    plt.title("SVM_SMO_LinearKernel")
    plt.show()

if __name__=='__main__':
    dataArr,classLabels = loadDataSet('testSet.txt') #读取数据集
    b,alphas = smoP(dataArr,classLabels,0.6,0.001,40) #基于给定参数，采用SMO算法对数据集进行划分
    w = calcWs(alphas,dataArr,classLabels) #基于参数矩阵与数据矩阵计算划分超平面模型系数
    showClassifer(dataArr,classLabels,w,b) #分类结果可视化        
        
             


        
                
        
        
        
    

        
             
                

            