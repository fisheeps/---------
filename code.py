
# coding: utf-8

# # Question 1 (60%, recognizing digit in an image using Logistic Regression model)
# 
# In this question, we are going to train a logistic regression model to recognize the digit in an image. The images used are mnist hand-written digit. The image is with size 28 by 28. Each image is vectorized (concatenate columns in the image sequentially) as a vector with length 784. The set of labels is {0,1,2,3,4,5,6,7,8,9}. We only pick two labels for this question so that the classification problem is a binary classification which logistic regression is able to deal with.
# 
# There are 60,000 training samples and 10,000 testing samples.
# 
# The following python commands may help you in solving the problem:
# 
# reshape, np.tile, np.vstack, np.hstack, np.concatenate, np.clip
# 
# > 1. Please note that for-loop and while-loop are not encouraged to use due to the low efficiency. Code with for-loop or while-loop would lose marks.
# >
# > 2. Only write the code in the required blocks shown below
# >        ### write your code below ###
# >        
# >        ### write your code above ###
# Code written outside the block may lose marks.
# >
# > 3. Other parts of the code, including method name, input arguments, return values are prohibited to change.
# > 4. Please note that for-loop and while-loop are not encouraged to use due to the low efficiency. Code with for-loop or while-loop would lose marks.
# > 5. When exponential function is computed, such as in computing value of sigmoid function, tricks must be implemented to avoid stack overflow.
# > 6. Variable name should be concise, representing the variable, neither too long nor too short. Messy variable names may lose marks.
# > 7. Necessary comments are required. Otherwise it is impossible for others to read the code. Code that are hard to read and missing comments may lose marks.
# > 8. In course slides, sample input is a column vector, while in the code implementation sample input is a row vector. The purpose is to make the use of training input matrix conveniently. Thus, you need to change formulas in the course slides a bit following which to write the code.
# 

# In[3]:

import numpy as np
import struct
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.utils import gen_batches
np.random.seed(2022)

train_image_file = './MNIST/raw/train-images-idx3-ubyte'
train_label_file = './MNIST/raw/train-labels-idx1-ubyte'
test_image_file = './MNIST/raw/t10k-images-idx3-ubyte'
test_label_file = './MNIST/raw/t10k-labels-idx1-ubyte'


def decode_image(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(-1, 784)
        images = np.array(images, dtype = float)
    return images

def decode_label(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        magic, n = struct.unpack('>II',f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
        labels = np.array(labels, dtype = float)
    return labels

def load_data() -> tuple:
    train_X = decode_image(train_image_file)
    train_Y = decode_label(train_label_file)
    test_X = decode_image(test_image_file)
    test_Y = decode_label(test_label_file)
    return (train_X, train_Y, test_X, test_Y)


# In[5]:

trainX, trainY, testX, testY = load_data()

# digit1 and digit2 are two digits used for binary classification. You could change their values.
digit1 = 1
digit2 = 7

idx = (trainY == digit1)+(trainY==digit2)
trainX, trainY = trainX[idx, :], trainY[idx]
num_train, num_feature = trainX.shape
idx = (testY == digit1)+(testY==digit2)
testX, testY = testX[idx, :], testY[idx]
num_test = testX.shape[0]

plt.figure(1, figsize=(20,5))
for i in range(4):
    idx = np.random.choice(range(num_train))
    plt.subplot(int('14'+str(i+1)))
    plt.imshow(trainX[idx,:].reshape((28,28)))
    plt.title('label is %d'%trainY[idx])
plt.show()

print('number of features is %d'%num_feature)
print('number of training samples is %d'%num_train)
print('number of testing samples is %d'%num_test)
trainY[np.where(trainY == digit1)], trainY[ np.where(trainY ==digit2)] = 0,1
testY[np.where(testY == digit1)], testY[np.where(testY == digit2)] = 0,1


# In[ ]:

class LogisticRegression():
    def __init__(self, num_feature: int, learning_rate: float) -> None:
        '''
        Constructor
        Parameters:
          num_features is the number of features.
          learning_rate is the learning rate.
        Return:
          there is no return value.
        '''
        self.num_feature = num_feature
        self.w = np.random.randn(num_feature + 1)
        self.learning_rate = learning_rate

    def artificial_feature(self, x: np.ndarray)->np.ndarray:
        '''
        add one artificial features to the data input
        Parameters:
          x is the data input. x is one dimensional or two dimensional numpy array.
        Return:
          updated data input with the last column of x being 1s.
        '''
        if len(x.shape) == 1: # if x is one dimensional, convert it to be two dimensional
            x = x.reshape((1, -1))
        #### write your code below ####
        if len(x.shape) == 1:
            X=np.c_([x,1])
        else:
            X=np.c_[x,np.ones(x.shape[0])]#这里在x矩阵的最后一列再添加一列1，因为类的成员变量w定义是feature+1维，说明把b这个
            #偏置也放入了参数向量当中一起参与后续计算了
        #### write yoru codel
        return X

    def sigmoid(self, x: np.ndarray)-> np.ndarray:
        '''
        Compute sigmoid activation function value by f(x*w)
        Parameters:
          x is data input with artificial features. x is a two dimensional numpy array.
        Return:
          one dimensional numpy array
        '''
        ### write your code below ###
        # first compute inner product between x and self.w
        # sencond, compute logistic function value of x*self.w
        prob=0.5+0.5*np.tanh(np.matmul(x,self.w)/2)#因为防止指数函数溢出，还不让用循环，只好使用tanh函数与sigmoid函数关系计算
        #正常的话，sigmoid输入大于0，输出1/(1+exp(-输入)),输入小于0，输出1/(1+exp(输入))，sigmoid(x)=0.5+0.5*tanh(x/2)
        ### write your code above ###
        belta=np.max(np.matmul(x,self.w))
        return prob

    def predict(self, X: np.ndarray)->np.ndarray:
        '''
        Predict label probability for the input X
        Parameters:
          X is the data input. X is one dimensional or two dimensional numpy array.
        Return: 
          predicted label probability, which is a one dimensional numpy array.
        '''
        X = self.artificial_feature(X)
        #### write your code below ####
        prob=self.sigmoid(X)#调用sigmoid接口直接计算概率
        #### write your code above ####
        return prob

    def loss(self, y: np.ndarray, prob: np.ndarray)->float:
        '''
        Compute cross entropy loss.
        Parameters:
          y is the true label. y is a one dimensional array.
          prob is the predicted label probability. prob is a one dimensional array.
        Return:
          cross entropy loss
        '''
        #### write your code below ####
        #### you must think of how to deal with the case that prob contains 1 or 0 ####
        delta = 1e-7#防止log(0)产生而溢出
        y=y.reshape(-1,1)
        prob=prob.reshape(-1,1)
        loss_value=-((y*np.log(prob+delta)+(np.ones(y.shape)-y)*np.log(1-prob+delta))).sum()#交叉熵损失定义
        #### write your code above ####
        return loss_value

    def gradient(self, trainX: np.ndarray, trainY: np.ndarray)->np.ndarray:
        '''
        Compute gradient of logistic regression.
        Parameters:
          trainX is the training data input. trainX is a two two dimensional numpy array.
          trainY is the training data label. trainY is a one dimensional numpy array.
        Return:
          a one dimensional numpy array representing the gradient
        '''
        x = self.artificial_feature(trainX)
        #### write your code below ####
        result=(self.sigmoid(x)-trainY).reshape(-1,1)#根据公式推导,假设X数据矩阵是一个N*(features+1)维矩阵,其中N代表送入训练的
        #样本个数,feature代表每个样本的维度(784)，+1是因为把常数项b也融合到了权重向量当中。交叉熵梯度=求和(第i个样本的特征向量(X矩阵的第i行)
        #的转置乘(第i个样本的预测概率减真实标签),i=1,2,....,训练样本个数)
        g=np.sum(np.multiply(x,result),axis=0)/len(trainY)#np.multiply是矩阵与向量对应相乘,例如x=[[1,2,3],[4,5,6]],y=[1,2,3]
        #np.multiply(x,y)=[[1*2,2*2,3*3],[4*1,5*2,6*3]]
        #### write your code above ####
        return g
    
    def update_weight(self, dLdw: np.ndarray)-> None:
        '''
        Update parameters of logistic regression using the given gradient.
        Parameters:
          dLdw is a one dimensional gradient.
        Return:
          there is no return value
        '''
        self.w += -self.learning_rate*dLdw
        return
    
    def one_epoch(self, X: np.ndarray,  Y: np.ndarray, batch_size: int, train : bool = True)->tuple:
        '''
        One epoch of either training or testing procedure.
        Parameters:
          X is the data input. X is a two dimensional numpy array.
          Y is the data label. Y is a one dimensional numpy array.
          batch_size is the number of samples in each batch.
          train is a boolean value indicating training or testing procedure.
        Returns:
          loss_value is the average loss function value.
          acc is the prediction accuracy.        
        '''
        num_sample = X.shape[0] # number of samples
        num_correct = 0        # number of corrected predicted samples
        num_batch = int(num_sample/batch_size)+1 # number of batch
        batch_index = list(gen_batches(num_sample, num_batch)) # index for each batch
        loss_value = 0 # loss function value
        for i, index in enumerate(batch_index): # the ith batch
            x, y = X[index,:], Y[index] # get a batch of samples
            if train:
                dLdw = self.gradient(x, y) # compute gradient
                self.update_weight(dLdw)   # update parameters of the model
            prob = self.predict(x)        # predict the label probability
            loss_value += self.loss(y, prob)*x.shape[0]  # loss function value for ith batch
            num_correct += self.accuracy(y, prob)*x.shape[0]
        loss_value = loss_value/num_sample # average loss
        acc = num_correct/num_sample       # accuracy
        return loss_value, acc
    
    def accuracy(self, y: np.ndarray, prob: np.ndarray)-> float:
        '''
        compute accuracy
        Parameters:
          y is the true label. y is a one dimensional array.
          prob is the predicted label probability. prob is a one dimensional array.
        Return:
          acc is the accuracy value
        '''
        acc=(((np.array(y)==1)==(np.array(prob)>=0.5)).sum())/len(y)#np.array(y)==1是为了获得一个bool数组,其中若y中
        #某元素为1，即真实标签为1，则该数组对应位置记True(代表正类)，否则记False(负类)。同理，prob经过处理也会得到一个bool数组,
        #将这两个数组对应位置比较判断，得到一个新的bool数组，数组的每个True元素意味着预测与真实标签是相同的，sum()用于求相同的个数
        #再除总长度即为准确率
        #### write your code above ####
        return  acc
    


# In[ ]:

def train(model, trainX, trainY, epoches, batch_size):
    loss_value, acc = model.one_epoch(trainX, trainY, batch_size, train = False)
    print('Initialization: ', 'loss %.4f  '%loss_value, 'accuracy %.2f'%acc)
    for epoch in range(epoches):
        loss_value, acc = model.one_epoch(trainX, trainY, batch_size)
        print('epoch: %d'%(epoch+1), 'loss %.4f  '%loss_value, 'accuracy %.2f'%acc)


# In[ ]:

model = LogisticRegression(num_feature, learning_rate = 0.01)
train(model, trainX, trainY, epoches = 10, batch_size = 256)


# In[ ]:

test_loss, test_acc = model.one_epoch(testX, testY, batch_size = 256, train = False)
print('testing accuracy is %.4f'%test_acc)

