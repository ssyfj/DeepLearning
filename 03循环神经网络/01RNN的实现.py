#!/usr/bin/env python
# coding: utf-8

# 详细见：https://www.zybuluo.com/hanbingtao/note/541458
# ====
# https://zhuanlan.zhihu.com/p/26892413

# In[14]:


import numpy as np


# In[32]:


#激活函数
class ReluActivator(object):
    def forward(self,input_): #激活函数
        return max(0,input_)

    def backward(self,output): #导函数
        return 1 if output > 0 else 0
    
class IdentityActivator(object): #f(x) = x 后面梯度检验的时候使用了
    def forward(self,input_):
        return input_
    
    def backward(self,output):
        return 1


# In[16]:


#补充函数：我们的激活函数是针对单个值，而我们的数据是多维，所以我们需要一个函数进行数据转换处理
def element_wise_op(array,op):
    for i in np.nditer(array,op_flags=["readwrite"]): #数据单个迭代,无论几维，都处理为单个数据处理
        i[...] = op(i)  #[...]表示对原始数据进行修改


# 一：循环神经网络
# =====
#   
# ![image.png](attachment:image.png)
#   
# 从后面代码可以知道：其实所有部分都是共用权值W,U的，只不过每一次权值的更新，前面的权值都会产生影响
#   
#   
# 训练算法BPTT
# ----
#   
# ![image-2.png](attachment:image-2.png)

# 二：前向传播
# ======
#   
# ![image.png](attachment:image.png)
#   
#   
# 下面的讲解都是针对这一个时刻进行的。下面的所有向量、矩阵都是针对上图展开的
# ----
#   
# ![image-2.png](attachment:image-2.png)
#     
# ![image-3.png](attachment:image-3.png)

# 三：误差项的计算
# =====
# 注意：我们所有求解的误差项值都是针对时刻t的（当前时刻），虽然下面公式会涉及到前面其他时刻，但是都是为了辅助当前时刻而求解的。如下：涉及时间的会辅助获取前面时间刻度下的误差，而在层级（不涉及时间的），我们只需要求解当前时刻t下的l-1层误差即可。
# ----
#   
# ![image.png](attachment:image.png)
#     
# 每一步的误差项都是有用的，用来后面更新权值梯度。
#   
#   
# （一）沿时间轴往前传递一个时刻（t->t-1)时的规律,求解误差项
# -----
#   
# ![image-8.png](attachment:image-8.png)
#   
# ![image-9.png](attachment:image-9.png)
#   
# ![image-2.png](attachment:image-2.png)
#   
# ![image-3.png](attachment:image-3.png)
#   
# ![image-4.png](attachment:image-4.png)
#   
# ![image-5.png](attachment:image-5.png)
#   
# ![image-6.png](attachment:image-6.png)
#    
# 
# （二）神经网络层级之间的误差传递(l->l-1)
# -----
#   
#   
# 与前面的时间刻度输出无关（注意），但是当前时刻的误差会传递到前面时刻下，对其他时刻下的l-1层产生影响。
#   
# ![image-11.png](attachment:image-11.png)
#   
# ![image-7.png](attachment:image-7.png)
# 
# （三）对比
# -----
# 时间刻度：
#   
# ![image-12.png](attachment:image-12.png)
#   
# 可以看出，每一个时间刻度下的误差值都与后一个时间刻度有关系 δ\<t-1> = δ\<t> x W x diag[f\`(net\<t-1>)]
# 注意：f\`(net\<t-1>)实际上是对第state\<t-1>调用激活对象的backward方法
#   
# 层级：
#   
# ![image-13.png](attachment:image-13.png)
#   
# 
# 可以看出，我们应该先求出时间刻度上δ值，然后求解前面层级的误差。注意：求解的只与当前t时刻有关，与前面时刻的输出无关。
# ----

# 四：权重梯度的计算
# ====
#   
# 当前时刻是t=6时：
#   
# ![image-9.png](attachment:image-9.png)
#   
# ![image-6.png](attachment:image-6.png)
#   
# ![image-7.png](attachment:image-7.png)
#   
# ![image-8.png](attachment:image-8.png)

# ![image.png](attachment:image.png)
#   
# ![image-2.png](attachment:image-2.png)

# 五：实现RNN
# ====
# ![image.png](attachment:image.png)

# In[47]:


class RecurrentLayer(object):
    def __init__(self,input_width,state_width,activator,learning_rate):
        """
        根据state_width可以知道中间隐藏层的单元大小，并且知道权重W是（state_width,state_width);
        根据input_width可以知道输入层输入单元个数，从而知道权重U是（input_width,state_width)
        """
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        
        self.times = 0 #当前时刻初始化为0,用来记录时间
        self.state_list = [] #保存各个时刻的state
        self.state_list.append(np.zeros((state_width,1))) #初始化s0
        
        #从代码可以知道：其实所有部分都是共用权值的，只不过每一次权值的更新，前面的权值都会产生影响
        self.U = np.random.uniform(-1e-4,1e-4,(state_width,input_width)) #一般权值反向设置，方便之后梯度求解
        self.W = np.random.uniform(-1e-4,1e-4,(state_width,state_width))
        
        self.input_array = [] #记录各个时刻下的输入值
        self.input_array.append(np.zeros((input_width,1))) #把0时刻填充了，没用
        
    def forward(self,input_array): #注意：数据是列向量
        """
        根据前面前向传播式2,进行计算（一次计算一个）
        """
        self.times += 1 #s1 s2 .... ===》 (s0 s1 s2 .. st)
        state = self.U@input_array + self.W@self.state_list[-1]
        element_wise_op(state,self.activator.forward)
        self.state_list.append(state)
        self.input_array.append(input_array)
        
    def backward(self,sensitivity_array):
        """
        实现BPTT算法
        sensitivity_array可以认为是当前时刻下的误差项
        """
        self.calc_delta(sensitivity_array)
        self.calc_gradient()
        
        
    def calc_delta(self,sensitivity_array):
        """
        计算误差项
        """
        self.delta_list = [] #用来保存各个时刻的误差项
        for i in range(self.times):
            self.delta_list.append(np.zeros((self.state_width,1))) #形状如项目state一致 
            #δ0 δ1 δ2 ... ===》 （δ0 δ1 ... δ(t-1))
        self.delta_list.append(sensitivity_array)  #δ0 δ1 ... δ(t-1) δt
        #开始从后往前迭代获取各个时刻的误差项
        for k in range(self.times - 1,0,-1): #我们只需要求解 δ(t-1) ... δ1 δ0 
            self.calc_delta_time_k(k)
            
        #self.calc_delta_level_l() 未实现
    
    def calc_delta_time_k(self,k):
        """
        先求解时间轴上的误差项，然后才能求解各个层级的误差
        根据k+1时刻的delta求解k时刻的delta
        """
        #先对state[k]求导
        state = self.state_list[k].copy()
        #print(state.shape)  (2,1) 我们要diag 需要变为1维数据 使用[:,0]即可
        element_wise_op(state,self.activator.backward)
        #print(self.delta_list[k+1].shape,self.W.shape,np.diag(state[:,0]).shape) #(2, 1) (2, 2) (2,2)
        self.delta_list[k] = (self.delta_list[k+1].T@(self.W@np.diag(state[:,0]))).T
        
    def calc_delta_level_l(self,l):
        """
        根据前面的时间刻度求出的delta,我们下面对l-1层进行误差求解：由于我们没有l-1层，前向传播也没有记录对应的l-1层输出
        所以不方便求导，跳过....
        """
        pass
    
    def calc_gradient(self):
        """
        求解梯度，见式5,式6
        """
        self.gradient_list_w = [] #保存各个时刻下W的权重梯度
        self.gradient_list_u = [] #保存各个时刻下U的权重梯度
        
        for t in range(self.times+1): #0 1 ... t
            self.gradient_list_w.append(np.zeros((self.state_width,self.state_width)))
            self.gradient_list_u.append(np.zeros((self.state_width,self.input_width)))
            
        #计算各个时刻下的梯度
        for t in range(self.times,0,-1): # t t-1 ... 1 因为存在s0
            self.calc_gradient_w(t)
            self.calc_gradient_u(t)
            
        #实际的梯度是各个时刻梯度之和
        self.gradient_w = np.add.reduce(self.gradient_list_w)
        self.gradient_u = np.add.reduce(self.gradient_list_u)
    
    def calc_gradient_w(self,t):
        """
        求解时刻t下的W的梯度。参考式5
        """
        gradient = self.delta_list[t]@self.state_list[t-1].T
        self.gradient_list_w[t] = gradient
    
    def calc_gradient_u(self,t):
        """
        求解时刻t下的U的梯度。参考式6
        """
        gradient = self.delta_list[t]@self.input_array[t].T
        self.gradient_list_u[t] = gradient
        
    def update(self):
        """
        权值更新
        """
        self.W -= self.learning_rate*self.gradient_w
        self.U -= self.learning_rate*self.gradient_u
    
    def reset_state(self):
        """
        状态重置，每一轮新的数据进行训练时，我们都需要重置一些状态，尤其是在前向传播中出现的那些数据
        """
        self.times = 0
        self.state_list = []
        self.input_array = []
        self.state_list.append(np.zeros((self.state_width,1)))
        self.input_array.append(np.zeros((self.input_width,1)))


# 六：测试前向传播和反向传播
# =====

# In[48]:


def test():
    l = RecurrentLayer(3, 2, ReluActivator(), 1e-3)
    x, d = data_set()
    l.forward(x[0])
    l.forward(x[1])
    l.backward(d)
    return l


# In[49]:


test()


# 七：梯度检测
# ====

# In[20]:


def data_set():
    x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x, d


# In[54]:


def gradient_check():
    """
    梯度检查
    """
    error_func = lambda X:X.sum()
    
    r1 = RecurrentLayer(3,2,IdentityActivator(),1e-3)
    
    #前向计算
    X,d  = data_set()
    r1.forward(X[0])
    r1.forward(X[1])
    
    #设置sensitivity map
    sensitivity_map = np.ones(r1.state_list[-1].shape,dtype=np.float64)
    
    #计算梯度
    r1.backward(sensitivity_map)
    
    #检查梯度
    epsilon = 1e-4
    for i in range(r1.W.shape[0]):
        for j in range(r1.W.shape[1]):
            r1.W[i,j] += epsilon
            r1.reset_state()
            r1.forward(X[0])
            r1.forward(X[1])
            err1 = error_func(r1.state_list[-1])  #W是时间轴上的，我们对state_list[-1]操作即可
            r1.W[i,j] -= 2*epsilon
            r1.reset_state()
            r1.forward(X[0])
            r1.forward(X[1])
            err2 = error_func(r1.state_list[-1])
            
            expect_grad_w = (err1-err2)/(2*epsilon)
            r1.W[i,j] += epsilon
            print("Weights ---> W(%d,%d): expected:%f - actural:%f = %f"%(i,j,expect_grad_w,r1.gradient_w[i,j],expect_grad_w - r1.gradient_w[i,j]))

    for i in range(r1.U.shape[0]):
        for j in range(r1.U.shape[1]):
            r1.U[i,j] += epsilon
            r1.reset_state()
            r1.forward(X[0])
            r1.forward(X[1])
            err1 = error_func(r1.state_list[-1])  #W是时间轴上的，我们对state_list[-1]操作即可
            r1.U[i,j] -= 2*epsilon
            r1.reset_state()
            r1.forward(X[0])
            r1.forward(X[1])
            err2 = error_func(r1.state_list[-1])
            
            expect_grad_u = (err1-err2)/(2*epsilon)
            r1.U[i,j] += epsilon
            print("Weights ---> U(%d,%d): expected:%f - actural:%f = %f"%(i,j,expect_grad_u,r1.gradient_u[i,j],expect_grad_u - r1.gradient_u[i,j]))         


# In[55]:


gradient_check()

