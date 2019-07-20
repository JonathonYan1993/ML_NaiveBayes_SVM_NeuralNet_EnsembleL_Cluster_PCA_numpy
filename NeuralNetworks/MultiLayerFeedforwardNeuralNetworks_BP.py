# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:33:17 2019
本程序采用单隐层神经网络解决异或非线性可分问题，学习算法为BP算法
参考资料：
周志华《机器学习》第5章5.1～5.3节
github:https://github.com/Lupino/bpnn/blob/master/bpnn.py    
@author: yanji
"""

import math
import random 
import pickle

def sigmoid(x):
    """
    sigmoid函数选择双曲正切tanh函数，该函数比标准的对率函数好一点
    function:sigmoid函数，作为激活函数，依据输入与相应权重计算的X，计算神经元输出y
    Parameters: x - 函数输入
    Returns: 函数值
    """
    return math.tanh(x)
    #return 1.0/(1+math.exp(-x)) #对率函数

def dsigmoid(y):
    """
    function:计算激活函数的微分，即变量为x函数值为y时的导数，由于正切函数的特性只需用到函数值y就能求导
    Parameters: y - 激活函数值
    Returns: 激活函数的微分
    """
    return 1.0-y**2
    #return y*(1-y)

class Unit:
    """
    定义一个神经元的类，该类含有神经元对下一层各输出的权重，阈值可理解为下层额外输出为-1的权重，也会作为权重计算的一部分
    该类同时含有依据输入计算输出的函数、依据导数更新权重的函数、输出权重、设置权重的函数
    """
    def __init__(self,length):
        self.weight = [random.uniform(-0.2,0.2) for i in range(length)] #length为下一层输出的个数,随机初始化权重
        self.change = [0.0]*length  #初始化各权重的变化量
    def calc(self,sample):  #依据下层各输出，计算该神经元的输出
        self.sample=sample[:]
        partsum=sum([i*j for i,j in zip(self.sample,self.weight)]) #下层输出与权重对应相乘求和
        self.output = sigmoid(partsum)  #依据输入计算激活函数输出
        return self.output
    def update(self,diff,eta=0.5):
        #diff为Ek对该神经元激活函数自变量的导数，相当于《机器学习》西瓜书中的5.3节gj、eh的作用
        #eta为学习速率
        change = [-eta*x*diff for x in self.sample] #计算每个权重的变化量
        self.weight=[w+c for w,c in zip(self.weight,change)] #更新权重
    def get_weight(self):
        return self.weight[:]  #输出权重
    def set_weight(self,weight):
        self.weight=weight[:]  #设置权重

class Layer:
    """
    定义一个神经网络层，由各个神经单元组成，含有神经单元的个数（本层的输出）及下一层的输出
    本层的输出为各个神经元输出组成的向量，权重更新则对各个神经元进行权重更新，同时还要计算用于下层导数计算的本层对应的导数传递
    """
    def __init__(self,input_length,output_length):
        #input_length为底下一层的输出，output_length为本层的输出
        self.units = [Unit(input_length) for i in range(output_length)] #初始化本层每个神经元
        self.output = [0.0]*output_length #初始化本层输出
        self.ilen = input_length
    def calc(self,sample):
        self.output = [unit.calc(sample) for unit in self.units] #计算每个神经元的输出，组成本层输出向量
        return self.output[:]
    def update(self,diffs,eta=0.5): #diffs为Ek对各神经元对应自变量导数的组成向量
        for diff,unit in zip(diffs,self.units):
            unit.update(diff,eta)  #对每个神经元的权重进行更新
    def get_dLayer(self,deltas):
        #计算本层对底下一层的导数传递，相当于《机器学习》西瓜书中5.3节式5.15中sum(WhjGj)的作用
        def _dlayer(deltas,j): #计算本层对底下一层单个神经元的导数传递
            return sum([delta*unit.weight[j] for delta,unit in zip(deltas,self.units)])
        return [_dlayer(deltas,j) for j in range(self.ilen)] #返回对底下一层各个神经元的导数传递
    def get_weights(self):             #提取本模型各神经元权重，以便存储
        weights={}
        for key,unit in enumerate(self.units):
            weights[key]=unit.get_weight()
        return weights
    def set_weights(self,weights):     #利用已有模型设置各神经元权重
        for key,unit in enumerate(self.units):
            unit.set_weight(weights[key])

class BPNNet:
    #定义一个多层前馈神经网络类，包括神经网络的输出计算、权重更新、训练及测试
    def __init__(self,ni,nh,no):
        #ni为输入层输入个数，nh为隐藏层神经元个数，no为输出层神经元节点个数
        self.ni = ni+1 #加1是把权重对应的系数-1作为一个输入
        self.nh = nh
        self.no = no
        self.hlayer = Layer(self.ni,self.nh)  #初始化隐藏层
        self.olayer = Layer(self.nh,self.no)  #初始化输出层
    def calc(self,inputs):
        #依据输入计算神经网络输出
        if len(inputs)!=self.ni-1:
            raise ValueError('wrong number of inputs')
        #输入层
        self.ai = inputs[:] + [-1.0]
        #计算隐藏层输出
        self.ah = self.hlayer.calc(self.ai)
        #计算输出层输出
        self.ao = self.olayer.calc(self.ah)
        return self.ao[:]
    def update(self,targets,eta):
        #对整个神经网络的权重进行更新
        if len(targets)!=self.no:
            raise ValueError('wrong number of target values')
        #计算输出层的各神经元的导数
        output_deltas = [dsigmoid(ao)*(ao-target) for target,ao in zip(targets,self.ao)]
        #计算隐藏层的各神经元的导数，这需要用到输出层对隐藏层的导数传递
        hidden_deltas = [dsigmoid(ah)*dlayer for ah,dlayer in zip(self.ah,self.olayer.get_dLayer(output_deltas))]
        #更新输出层权重
        self.olayer.update(output_deltas,eta)
        #更新隐藏层权重
        self.hlayer.update(hidden_deltas,eta)
        #计算误差Ek
        return sum([0.5*(t-o)**2 for t,o in zip(targets,self.ao)])  #返回该样本的目前的预测误差
    def test(self,patterns):
        #测试样例
        for p in patterns:
            print(p[0],'真实结果:',p[1],'预测结果:',self.calc(p[0]))
    def train(self,patterns,iterations=1000,eta=0.5):
        #eta-学习速率
        for i in range(iterations): #遍历训练集，遍历总轮数为iterations
            error=0.0  #初始化本轮训练误差
            for p in patterns:  #遍历样本集
                inputs=p[0]
                targets=p[1]
                self.calc(inputs) #计算该样本的输出
                error = error + self.update(targets,eta) #对神经网络进行更新，并返回该样本的误差，累加到本轮训练误差
            if i%100 ==0:  #如果轮数是100的整数倍，输出该轮训练误差
                print('error:%.10f'%error)
    def save_weights(self,fn):  #将神经网络模型权重写入到文件中进行保存
        weights={
                'olayer':self.olayer.get_weights(),
                'hlayer':self.hlayer.get_weights()}
        with open(fn,'wb') as f:
            pickle.dump(weights,f)  #写入文件
    def load_weights(self,fn): #从已有文件中读取模型权重，添加到神经网络中
        with open(fn,'rb') as f:
            weights = pickle.load(f)  #读取文件
            self.olayer.set_weights(weights['olayer'])
            self.hlayer.set_weights(weights['hlayer'])
            
if __name__=='__main__':
    #建立异或数据矩阵，包括输入，输出
    pat=[
            [[0,0],[0]],
            [[0,1],[1]],
            [[1,0],[1]],
            [[1,1],[0]]
        ]
    #创建一个神经网络类，2个输入节点，2个隐藏神经元，一个输出
    n=BPNNet(2,2,1)
    #基于异或数据矩阵对神经网络进行训练
    n.train(pat)
    #保存训练模型的权重到文件中
    n.save_weights('xor.weights')
    #利用训练好的神经网络对异或数据矩阵进行测试
    n.test(pat)    
    
    