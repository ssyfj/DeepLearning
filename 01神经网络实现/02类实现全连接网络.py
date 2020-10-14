#!/usr/bin/env python
# coding: utf-8

# 讲解看01神经网络实现（含反向传播和梯度下降）.ipynb和https://www.zybuluo.com/hanbingtao/note/476663
#   
# 这里将前面函数整合为类进行处理

# In[185]:


import numpy as np


# 一：实现激活类SigmoidActivator
# ======

# In[186]:


class SigmoidActivator(object):
    #实现sigmoid函数
    def forward(self,Z): #之所以用forward和backward,是为了兼容多个不同的激活项对象
        return 1/(1+np.exp(-Z))
    
    #下面进行反向传播
    #各种函数求导：平方损失、交叉熵、多累分类https://zhuanlan.zhihu.com/p/99923080
    #sigmoid求导
    def backward(self,output): #注意：这里我们在前向传播中获取了输出值，不需要再次计算
        return output*(1-output)


# 二：实现全连接类
# ====

# In[ ]:


class FullConnectedLayer(object):
    def __init__(self,input_size,output_size,activator):
        """
        input_size:输入向量维度(激活单元数量)
        output_size：输出层向量维度(激活单元数量)
        activator：激活函数
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        
        #根据上面构造权值信息
        self.W = np.random.normal(-0.1,0.1,(output_size,input_size))
        #self.W = np.ones((output_size,input_size))
        self.b = np.zeros((output_size,1))
        
        #输出向量大小
        self.output = np.zeros((output_size,1))
    
    #注意下面前向传播和反向传播结合全链接层，前向传播获取输出层的激活项值，所以默认的我们的input也应该是上一个全链接层的输出激活项值

    def forward(self,input_array): #实现前向传播
        self.input = input_array
        self.output = self.activator.forward(self.input@self.W.T + self.b.T)
        
    def backward(self,delta_array): #实现反向传播
        """
        delta_array是输出层之后的误差项
        """
        #注意：在执行backward之前，我们一定确保执行了forward,获取了input和output
        #print(delta_array.shape,self.W.shape,self.output.shape)
        self.delta =  self.activator.backward(self.input)*(delta_array@self.W)#我们求得delta是输入层的误差项
        #上面求解的self.delta是为上一个全链接层做准备，我们求解梯度，至于本层的delta有关
        #print(delta_array.shape,self.output.shape)
        #print(delta_array.T.shape,self.input.shape)
        #print("----------")
        self.W_grad = delta_array.T@self.input
        self.b_grad = delta_array.T
        
    def update(self,learning_rate):
        self.W += learning_rate*self.W_grad
        #print(self.b.shape,self.b_grad.shape)
        #print(0000000000)
        self.b += learning_rate*self.b_grad
        
    def dump(self): #输出每一次的W和b信息
        print("W and b:")
        print(self.W,self.b)
        print("============")


# 三：神经网络构建类
# =======

# In[329]:


class Network(object):
    def __init__(self,layers):
        self.layers = []
        for i in range(len(layers)-1):
            self.layers.append(FullConnectedLayer(layers[i],layers[i+1],SigmoidActivator()))
    
    def predict(self,X): #预测函数（实际就是对整个网络实现了一次前向传播）
        output = X
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output
    
    def calc_gradient(self,label,y_pred):
        #先求出最后输出层的delta项
        delta = self.layers[-1].activator.backward(y_pred)*(y_pred-label)
        #遍历每一层，获取每一层的delta保存在原有数据结构中
        for layer in self.layers[::-1]:
            layer.backward(delta) #backward会将数据保存在layer.delta中，并且获取了每一层的梯度信息
            delta = layer.delta
            
    def update_weight(self,learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)
    
    def train_one_sample(self,label,sample,learning_rate): #训练一个样本
        y_pred = self.predict(sample)
        print(self.loss(y_pred,label))
        self.calc_gradient(label,y_pred)
        self.update_weight(learning_rate)
    
    def train(self,labels,data_set,learning_rate,epoch): #训练函数
        for i in range(epoch):
            print("==============================")
            for d in range(len(data_set)):
                self.train_one_sample(np.array([labels[d]]),np.array([data_set[d]]),learning_rate)
                
    def dump(self):
        for layer in self.layers:
            layer.dump()
            
    def loss(self,output,label):
        return np.sum(1/2*(np.power(output-label,2)))
    
    def gradient_check(self,sample,label): #梯度检测
        #反向传播下的梯度获取
        y_pred = self.predict(sample)
        #print(y_pred,label)
        #print(y_pred.shape,label.shape)
        self.calc_gradient(label,y_pred)
        
        #求导情况下的导数
        epsilon = 1e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i,j] += epsilon
                    output = self.predict(sample)
                    err1 = self.loss(output,label)
                    
                    fc.W[i,j] -= 2*epsilon
                    output = self.predict(sample)
                    err2 = self.loss(output,label)
                    
                    expect_grad = (err1 - err2)/(2*epsilon)
                    
                    #还原权值
                    fc.W[i,j] += epsilon
                    print("Weights(%d %d) expected:%f actual:%f expect - actual = :%f"%(i,j,expect_grad,fc.W_grad[i,j],expect_grad-fc.W_grad[i,j]))
                    


# In[326]:


data_set = np.array([[1,2,1,1,1,1,1,0]])
labels = np.array([[1,0,0,0,0,0,0,0]])
net = Network([8, 3, 8])

net.gradient_check(data_set,labels)

