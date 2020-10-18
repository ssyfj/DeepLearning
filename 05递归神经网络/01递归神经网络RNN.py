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

#一：定义树节点结构，用来保存由递归神经网络生成的树
class TreeNode(object):
    def __init__(self,data,children=[],children_data=None):
        self.parent = None
        self.data = data #保存节点的数据（由前向传播计算得）
        self.delta = None #保持节点的误差项值（由反向传播计算得）
        self.children = children
        self.children_data = children_data #直接保存子节点链接好的数据，以后就不用二次操作（前向）
        
        for child in children:
            child.parent = self


#二：递归神经网络实现
class RecursiveLayer(object):
    def __init__(self,node_width,child_count,activator,learning_rate):
        """
        注意：各个结点的维度是一样的
        """
        self.node_width = node_width
        self.child_count = child_count
        self.activator = activator
        self.learning_rate = learning_rate
        
        #设置权重矩阵大小 ---> 注意：是由子节点向父节点前向传播的（决定下面权值矩阵形状）
        self.W = np.random.uniform(-1e-4,1e-4,(node_width,node_width*child_count))
        #设置偏置项
        self.b = np.zeros((node_width,1))
        #递归神经网络生成树的根节点
        self.root = None

    def forward(self,*children):
        """
        *children表示列表
        """
        #先将所有子节点的数据连接起来为一个向量
        children_data = self.concatenate(children)
        #前向计算获取父节点数据
        parent_data = self.activator.forward(np.dot(self.W,children_data)+self.b)
        #设置树节点
        self.root = TreeNode(parent_data,children,children_data)
        
    def concatenate(self,tree_nodes):
        """
        将所有子节点的数据按列链接
        """
        concat = np.zeros((0,1)) #需要使得连接的形状一致，因为是列扩展，所以一定是(0,1) 0表示空，但是现状为1列
        for node in tree_nodes:
            concat = np.r_[concat,node.data] #每一个节点数据都是（n,1)的列向量，我们扩展连接使用r_变为(n*m,1)
        return concat
        
        
    def backward(self,parent_delta):
        """
        BPTS反向传播算法
        """
        self.calc_delta(parent_delta,self.root)
        self.W_grad,self.b_grad = self.calc_gradient(self.root)
        
    def calc_delta(self,parent_delta,parent_node):
        """
        根据式2,计算每个子节点的delta
        """
        parent_node.delta = parent_delta #更新节点的误差值
        #开始计算每个子节点的delta
        if parent_node.children:
            #式2
            children_delta = np.dot(self.W.T,parent_delta)*self.activator.backward(parent_node.children_data)
            #开始分片给每个节点,计算索引（子节点编号，delta起始，delta结束）
            slices = [(i,i*self.node_width,(i+1)*self.node_width) for i in range(self.child_count)]
            #针对每个子节点，进行递归操作
            for s in slices:
                self.calc_delta(children_delta[s[1]:s[2]],parent_node.children[s[0]])
                
    def calc_gradient(self,parent_node):
        """
        计算每个节点权重的梯度，并将它们求和，得到最终的梯度 式3 式5
        """
        W_grad = np.zeros(self.W.shape)
        b_grad = np.zeros(self.b.shape)
        if not parent_node.children:
            return W_grad,b_grad
        
        node_W_grad = np.dot(parent_node.delta,parent_node.children_data.T)
        node_b_grad = parent_node.delta
        
        W_grad += node_W_grad
        b_grad += node_b_grad
        
        for child in parent_node.children:
            node_W_grad,node_b_grad = self.calc_gradient(child)
            W_grad += node_W_grad
            b_grad += node_b_grad
        
        return W_grad,b_grad
    
    def update(self):
        """
        更新权值
        """
        self.W -= self.learning_rate*self.W_grad
        self.b -= self.learning_rate*self.b_grad
        
    def reset_state(self):
        """
        每一次梯度检测，都要初始化状态
        """
        self.root = None
        
    def dump(self, **kwArgs):
        print('root.data: %s' % self.root.data)
        print('root.children_data: %s' % self.root.children_data)
        if 'dump_grad' in kwArgs:
            print('W_grad: %s' % self.W_grad)
            print('b_grad: %s' % self.b_grad)


#三：测试前向和反向传播
def data_set():
    children = [
        TreeNode(np.array([[1],[2]])),
        TreeNode(np.array([[3],[4]])),
        TreeNode(np.array([[5],[6]]))
    ]
    d = np.array([[0.5],[0.8]])
    return children, d

def test():
    children, d = data_set()
    rnn = RecursiveLayer(2, 2, IdentityActivator(), 1e-3)
    rnn.forward(children[0], children[1])
    rnn.dump()
    rnn.forward(rnn.root, children[2])
    rnn.dump()
    rnn.backward(d)
    rnn.dump(dump_grad='true')
    return rnn


test()


#四：梯度检测
def gradient_check():
    error_func = lambda X:X.sum()
    
    rnn = RecursiveLayer(2,2,IdentityActivator(),1e-3)
    
    X,d = data_set()
    rnn.forward(X[0],X[1]) #x[0] x[1] 生成本次父节点保存在rnn.root中
    rnn.forward(rnn.root,X[2]) #rnn.root X[2] 生成最终父节点保存在rnn.root中
    
    #设置人工误差项
    sensitivity_map = np.ones((rnn.node_width,1),dtype=np.float64)
    
    #反向传播计算梯度
    rnn.backward(sensitivity_map)
    
    
    #导数定义求解梯度
    epsilon = 1e-3
    for i in range(rnn.W.shape[0]):
        for j in range(rnn.W.shape[1]):
            rnn.W[i,j] += epsilon
            rnn.reset_state()
            rnn.forward(X[0],X[1])
            rnn.forward(rnn.root,X[2])
            err1 = error_func(rnn.root.data)
            
            rnn.W[i,j] -= 2*epsilon
            rnn.reset_state()
            rnn.forward(X[0],X[1])
            rnn.forward(rnn.root,X[2])
            err2 = error_func(rnn.root.data)
            
            expect_grad = (err1-err2)/(2*epsilon)
            rnn.W[i,j] += epsilon
            print("Weight(%d,%d): expect:%.4e - actural:%.4e = %.4e"%(i,j,expect_grad,rnn.W_grad[i,j],expect_grad - rnn.W_grad[i,j]))


gradient_check()


