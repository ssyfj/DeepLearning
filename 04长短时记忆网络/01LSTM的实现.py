import numpy as np

#一：激活函数的实现
class SigmoidActivator(object):
    def forward(self,X):
        return 1/(1+np.exp(-X))
    
    def backward(self,Z):
        return Z*(1-Z)
    
class TanhActivator(object):
    def forward(self,X):
        return 2/(1+np.exp(-2*X))-1 #式10,同乘e^-z,然后+1,入分子，后-1即可
    
    def backward(self,Z):
        return 1-Z*Z
    
class IdentityActivator(object): #用来验证梯度检测
    def forward(self,input_):
        return input_
    
    def backward(self,output):
        return 1


class LstmLayer(object):
    def __init__(self,input_width,state_width,learning_rate):
        """
        前向传播6个，按时间序列初始
        遗忘门、输入门、当前单元状态、输出门与输入数据之间权值矩阵初始化
        """
        self.input_width = input_width
        self.state_width = state_width
        self.learning_rate = learning_rate
        
        #激活函数初始化，门激活函数Sigmoid 输出激活函数tanh
        self.gate_activator = SigmoidActivator()
        self.output_activator = TanhActivator()
        
        self.times = 0 #时刻记录
        self.input_array = []
        self.input_array.append(np.zeros((input_width,1))) #初始化时刻0下的输入数据
        
        #开始初始化前向传播中6个列表（式1--->式6） 各个向量都是同state_width大小，所以可以独立出一个函数
        #遗忘门
        self.f_list = self.init_state_vec()
        #输入门
        self.i_list = self.init_state_vec()
        #当前输入的单元状态
        self.c_list = self.init_state_vec()
        #当前时刻的单元状态
        self.ct_list = self.init_state_vec()
        #输出门
        self.o_list = self.init_state_vec()
        #输出结果
        self.h_list = self.init_state_vec()
        
        #开始初始化权重信息,由三可以知道，我们需要将每一个权值拆分为3类Wh,Wx,b分别针对上一个时刻的输出，这一个时刻的输入，以及偏移值
        #其中，针对输入门，遗忘门，当前输入的单元状态，输出门，各有一个权值。但是权值之间有共同性，所以我们可以独立一个函数初始化权值矩阵
        #遗忘门
        self.Wfh,self.Wfx,self.bf = self.init_weight_mat()
        #输入门
        self.Wih,self.Wix,self.bi = self.init_weight_mat()
        #当前时刻单元状态权重矩阵
        self.Wch,self.Wcx,self.bc = self.init_weight_mat()
        #输出门
        self.Woh,self.Wox,self.bo = self.init_weight_mat()        
        
    def init_state_vec(self):
        state_vec_list = []
        state_vec_list.append(np.zeros((self.state_width,1))) #将0时刻的数据填充即可
        return state_vec_list
    
    def init_weight_mat(self):
        Wh = np.random.uniform(-1e-4,1e-4,(self.state_width,self.state_width))
        Wx = np.random.uniform(-1e-4,1e-4,(self.state_width,self.input_width))
        b = np.zeros((self.state_width,1)) #注意：偏置值是针对上面两个Wh,Wx的
        return Wh,Wx,b
    
    def forward(self,X):
        """
        按照四中：式1到式6进行前向计算
        """
        self.times += 1 #时刻向前加1
        
        self.input_array.append(X) #添加输入数据到列表中
        
        #其中 遗忘门、输入门、当前输入的单元状态、输出门之间有共同性，独立一个函数
        #遗忘门
        ft_g = self.calc_gate(X,self.Wfx,self.Wfh,self.bf,self.gate_activator)
        self.f_list.append(ft_g)
        
        #输入门
        it_g = self.calc_gate(X,self.Wix,self.Wih,self.bi,self.gate_activator)
        self.i_list.append(it_g)
        
        #当前输入的单元状态
        ct = self.calc_gate(X,self.Wcx,self.Wch,self.bc,self.output_activator)
        self.ct_list.append(ct)
        
        #当前时刻输出的单元状态
        c = ft_g*self.c_list[self.times - 1] + it_g*ct
        self.c_list.append(c)
        
        #输出门
        ot_g = self.calc_gate(X,self.Wox,self.Woh,self.bo,self.gate_activator)
        self.o_list.append(ot_g)
        
        #输出结果
        h = ot_g*self.output_activator.forward(c)
        self.h_list.append(h)        
        
        
    def calc_gate(self,x,Wx,Wh,b,activator):
        """
        计算门 h<t-1>是不变的，所以不用设置在传参中
        """
        h_pre = self.h_list[self.times - 1] #上一个时刻的输出
        net = np.dot(Wh,h_pre) + np.dot(Wx,x) + b
        return activator.forward(net)
    
    #开始反向传播的实现
    def backward(self,sensitivity_array):
        """
        计算误差项，为了方便梯度检测测试，
        我们这里自己提供最后t时刻的误差项sensitivity_array。
        实际中，应该使用前向传播结果进行推导
        """
        self.calc_delta(sensitivity_array)
        self.calc_gradient()
        
    def calc_delta(self,sensitivity_array):
        #首先：我们应该初始化各个误差项的数据结构
        #输出误差项 delta_h
        self.delta_h_list = self.init_delta()
        #输出门误差项
        self.delta_o_list = self.init_delta()
        #输入门误差项
        self.delta_i_list = self.init_delta()
        #遗忘门误差项
        self.delta_f_list = self.init_delta()
        #当前输入的单元状态误差项
        self.delta_ct_list = self.init_delta()
        
        #将参数中的输出误差项加入delta_h_list
        self.delta_h_list[-1] = sensitivity_array
        
        #开始递归时间求解各个时间刻度下的误差项
        for k in range(self.times,0,-1): #s<t> s<t-1> ... s2 s1
            self.calc_delta_time_k(k)
            
        #因为我们的网络中只有一个输入层，隐藏层，输出层，不需要求解层级之间误差项
        
    def init_delta(self):
        delta_list = []
        for i in range(self.times+1): #s0 s1 ... s<t>
            delta_list.append(np.zeros((self.state_width,1)))
            
        return delta_list
    
    def calc_delta_time_k(self,k):
        """
        计算各个时间刻度下的误差项（误差项沿时间的反向传递），参考五1中式8-->式13
        """
        delta_k = self.delta_h_list[k]
        fg = self.f_list[k]
        ig = self.i_list[k]
        og = self.o_list[k]
        ct = self.ct_list[k]
        c = self.c_list[k]
        c_prev = self.c_list[k-1]
        
        tanh_c = self.output_activator.forward(c)
        tanh_pow_c = 1-tanh_c*tanh_c
        
        #根据五1中式8-->式13计算
        delta_o = (delta_k*tanh_c*self.gate_activator.backward(og))
        delta_f = (delta_k*og*tanh_pow_c*c_prev*self.gate_activator.backward(fg))
        delta_i = (delta_k*og*tanh_pow_c*ct*self.gate_activator.backward(ig))
        delta_ct = (delta_k*og*tanh_pow_c*ig*self.output_activator.backward(ct))
        
        delta_h_prev = (np.dot(delta_o.T,self.Woh)+np.dot(delta_f.T,self.Wfh)+
                       np.dot(delta_i.T,self.Wih)+np.dot(delta_ct.T,self.Wch)).T
        
        #将上面的数据全部保存
        self.delta_o_list[k] = delta_o
        self.delta_i_list[k] = delta_i
        self.delta_f_list[k] = delta_f
        self.delta_ct_list[k] = delta_ct
        
        self.delta_h_list[k-1] = delta_h_prev
        
        
    def calc_gradient(self):
        """
        计算梯度，参考六 77-80 92-95 96-105
        """
        #遗忘门
        self.Wfh_grad,self.Wfx_grad,self.bf_grad = self.init_weight_gradient_mat()
        #输入门
        self.Wih_grad,self.Wix_grad,self.bi_grad = self.init_weight_gradient_mat()
        #当前时刻单元状态权重矩阵
        self.Wch_grad,self.Wcx_grad,self.bc_grad = self.init_weight_gradient_mat()
        #输出门
        self.Woh_grad,self.Wox_grad,self.bo_grad = self.init_weight_gradient_mat() 
        
        #按时间更新梯度信息，进行累加到上面全局梯度中
        for t in range(self.times,0,-1):
            (wfh_grad,bf_grad,
            wih_grad,bi_grad,
            wch_grad,bc_grad,
            woh_grad,bo_grad) = (self.calc_gradient_t(t))
            
            self.Wfh_grad += wfh_grad
            self.bf_grad += bf_grad
            
            self.Wih_grad += wih_grad
            self.bi_grad += bi_grad
            
            self.Wch_grad += wch_grad
            self.bc_grad += bc_grad
            
            self.Woh_grad += woh_grad
            self.bo_grad += bo_grad
            
        #计算层级方向的权重梯度
        self.Wfx_grad = np.dot(self.delta_f_list[-1],self.input_array[-1].T)
        self.Wix_grad = np.dot(self.delta_i_list[-1],self.input_array[-1].T)
        self.Wcx_grad = np.dot(self.delta_ct_list[-1],self.input_array[-1].T)
        self.Wox_grad = np.dot(self.delta_o_list[-1],self.input_array[-1].T)
        
    def init_weight_gradient_mat(self):
        """
        不同于上面初始化权重矩阵，虽然两个的形状一样，但是我们这里填充的数据为0,上面是随机初始化
        """
        Wh = np.zeros((self.state_width,self.state_width))
        Wx = np.zeros((self.state_width,self.input_width))
        b = np.zeros((self.state_width,1))
        return Wh,Wx,b
    
    def calc_gradient_t(self,k):
        """
        按时间计算权值梯度信息，参考六 77-80 92-95 96-105
        """
        h_prev = self.h_list[k-1].T
        
        wfh_grad = np.dot(self.delta_f_list[k],h_prev)
        bf_grad = self.delta_f_list[k]
        
        wih_grad = np.dot(self.delta_i_list[k],h_prev)
        bi_grad = self.delta_i_list[k]
        
        woh_grad = np.dot(self.delta_o_list[k],h_prev)
        bo_grad = self.delta_o_list[k]
        
        wch_grad = np.dot(self.delta_ct_list[k],h_prev)
        bc_grad = self.delta_ct_list[k]
        
        return wfh_grad,bf_grad,wih_grad,bi_grad,wch_grad,bc_grad,woh_grad,bo_grad
    
    def update(self):
        """
        更新权值，根据上面的梯度信息
        """
        self.Wfh -= self.learning_rate*self.Wfh_grad
        self.Wih -= self.learning_rate*self.Wih_grad
        self.Wch -= self.learning_rate*self.Wch_grad
        self.Woh -= self.learning_rate*self.Woh_grad
        
        self.bf -= self.learning_rate*self.bf_grad
        self.bi -= self.learning_rate*self.bi_grad
        self.bc -= self.learning_rate*self.bc_grad
        self.bo -= self.learning_rate*self.bo_grad
        
        self.Wfx -= self.learning_rate*self.Wfx_grad
        self.Wix -= self.learning_rate*self.Wix_grad
        self.Wcx -= self.learning_rate*self.Wcx_grad
        self.Wox -= self.learning_rate*self.Wox_grad
        
    def reset_state(self):
        """
        和RecurrentLayer一样
        为了支持梯度检查
        状态重置，每一轮新的数据进行训练时，我们都需要重置一些状态，尤其是在前向传播中出现的那些数据
        """
        
        self.times = 0
        
        #遗忘门
        self.f_list = self.init_state_vec()
        #输入门
        self.i_list = self.init_state_vec()
        #当前输入的单元状态
        self.c_list = self.init_state_vec()
        #当前时刻的单元状态
        self.ct_list = self.init_state_vec()
        #输出门
        self.o_list = self.init_state_vec()
        #输出结果
        self.h_list = self.init_state_vec()
        
# 八：测试前向传播和反向传播
# ===

#测试数据
def data_set():
    x = [np.array([[1], [2], [3]]),
         np.array([[2], [3], [4]])]
    d = np.array([[1], [2]])
    return x, d #返回的d,用来作为最后时刻的误差项，测试反向传播


#测试前向传播和反向传播
def test():
    lstm = LstmLayer(3,2,1e-3)
    X,sensitivity_map = data_set()
    lstm.forward(X[0])
    lstm.forward(X[1])
    lstm.backward(sensitivity_map)


test()


# 九：实现梯度检测
# ===

def gradient_check():
    """
    实现梯度检测
    """
    error_func = lambda X:X.sum()
    
    lstm = LstmLayer(3,2,1e-3)
    
    #使用我们的反向传播获取梯度信息
    X,d = data_set()
    lstm.forward(X[0])
    lstm.forward(X[1])
    
    #设置我们自定义的t时刻的误差项（之前RNN是直接使用d)
    sensitivity_map = np.ones(lstm.h_list[-1].shape,dtype=np.float64)
    
    #反向传播，获取梯度信息
    lstm.backward(sensitivity_map)
    
    #开始实现导数定义求解梯度
    epsilon = 1e-3
    
    #由于时间刻度上有多个权值Wfh,Wih,Woh,Wch,我们这里获取一个查看
    for i in range(lstm.Wfh.shape[0]):
        for j in range(lstm.Wfh.shape[1]):
            lstm.Wfh[i,j] += epsilon
            
            lstm.reset_state()
            lstm.forward(X[0])
            lstm.forward(X[1])
            
            err1 = error_func(lstm.h_list[-1])
            
            lstm.Wfh[i,j] -= 2*epsilon
            
            lstm.reset_state()
            lstm.forward(X[0])
            lstm.forward(X[1])
            
            err2 = error_func(lstm.h_list[-1])
            
            expect_grad = (err1-err2)/(2*epsilon)
            lstm.Wfh[i,j] += epsilon #注意：需要还原回去
            
            print("Weights time ---> W(%d,%d): expected:%.4e - actural:%.4e = %.4e"%(i,j,expect_grad,lstm.Wfh_grad[i,j],expect_grad - lstm.Wfh_grad[i,j]))

    #同样：对层级方向的权值梯度进行一次检测
    for i in range(lstm.Wfx.shape[0]):
        for j in range(lstm.Wfx.shape[0]):
            lstm.Wfx[i,j] += epsilon
            
            lstm.reset_state()
            lstm.forward(X[0])
            lstm.forward(X[1])
            
            err1 = error_func(lstm.h_list[-1])
            
            lstm.Wfx[i,j] -= 2*epsilon
            
            lstm.reset_state()
            lstm.forward(X[0])
            lstm.forward(X[1])
            
            err2 = error_func(lstm.h_list[-1])
            
            expect_grad = (err1-err2)/(2*epsilon)
            lstm.Wfx[i,j] += epsilon #注意：需要还原回去
            
            print("Weights level ---> W(%d,%d): expected:%.4e - actural:%.4e = %.4e"%(i,j,expect_grad,lstm.Wfx_grad[i,j],expect_grad - lstm.Wfx_grad[i,j]))


gradient_check()

