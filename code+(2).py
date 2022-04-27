
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

# In[1]:

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


# In[2]:


# # Question 2 (40%,  recognizing the digit in an image using Multinomial Logistic Regression model)
# 
# In this questin, we are going to implement multinomial regression to recognize digit in mnist hand-written images. The number of classes is now ten. 
# 
# > 1. Please note that in course slides, sample input is a column vector, while in the code implementation sample input is a row vector. The purpose is to make the use of training input matrix conveniently. Thus, you need to change formulas in the course slides a bit following which to write the code.
# > 2. Labels are in the form of **one-hot vectors**.

# In[7]:

trainX, trainY, testX, testY = load_data()

num_train, num_feature = trainX.shape
plt.figure(1, figsize=(20,10))
for i in range(9):
    idx = np.random.choice(range(num_train))
    plt.subplot(int('33'+str(i+1)))
    plt.imshow(trainX[idx,:].reshape((28,28)))
    plt.title('label is %d'%trainY[idx])
plt.show()


# In[8]:

def to_onehot(y: np.ndarray)-> np.ndarray:
    '''
    Covert ordinal labels to one-hot vector labels.
    Parameters:
      y is a one dimensional numpy array containing oridinal class labels.
    Returns:
      a two dimensional numpy array with each row a one-hot vector.    
    '''
    y = y.astype(int)
    num_class = len(set(y))
    Y = np.eye((num_class))
    return Y[y]

trainY = to_onehot(trainY) # convert traing sample labels to one-hot vectors
testY = to_onehot(testY)  # convert testing sample labels to one-hot vectors
num_train, num_feature = trainX.shape
num_test, _ = testX.shape
_, num_class = trainY.shape
print('number of features is %d'%num_feature)
print('number of classes is %d'%num_class)
print('number of training samples is %d'%num_train)
print('number of testing samples is %d'%num_test)


# In[9]:

class MNRegression():
    def __init__(self, num_feature: int, num_class: int, learning_rate: float) -> np.ndarray:
        self.num_feature = num_feature
        self.num_class = num_class
        self.W = np.random.randn(num_feature + 1, num_class)
        self.learning_rate = learning_rate

    def artificial_feature(self, x: np.ndarray) -> np.ndarray:
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

        #X = np.insert(x,x.shape[1],1,axis=1)
        if len(x.shape) == 1:
            X=np.c_([x,1])
        else:
            X=np.c_[x,np.ones(x.shape[0])]
        #### write yoru codel
        return X
        
    def softmax(self, x:np.ndarray)->np.ndarray:
        '''
        softmax function
        Parameters:
          x is a two dimensional numpy array representing data input
        Returns:
          value of softmax function. which is a two dimensional numpy array.
        '''
        num_sample = x.shape[0]
        num_class = self.W.shape[1]
        #### write your code below ####
        # first compute x*self.W
        # second, compute softmax(x*self.W)
        result=np.matmul(x,self.W)
        max_value = np.max(result, axis=1)
        ones=np.ones(num_class)
        ones=ones.reshape(1,-1)
        max_value=max_value.reshape(-1,1)
        x_exp = np.exp(result-np.matmul(max_value,ones))
        partition = x_exp.sum(axis = 1, keepdims = True)
        prob = x_exp / partition
        #### write your code above ####
        return prob

    def predict(self, X: np.ndarray)-> np.ndarray:
        '''
        Predict label probability for the input X
        Parameters:
          X is the data input. X is one dimensional or two dimensional numpy array.
        Return: 
          predicted label probability, which is a two dimensional numpy array.
        '''
        x = self.artificial_feature(X)
        #### write your code below ####

        prob=self.softmax(x)#调用softmax接口直接计算概率

        #### write your code above ####
        return prob

    def loss(self, y: np.ndarray, prob: np.ndarray)->float:
        '''
        Compute the cross entropy loss
        Parameters:
          y is the true label, which is a two dimensional array.
          prob is the predicted label probability, which is a two dimensional array.
        Return:
          cross entropy loss, which is a scalar.
        
        NOTE that for each sample input the predicted label proability is a one dimensional array other than a number.
        '''
        #### write your code below ####

        delta = 1e-7#防止log(0)产生而溢出
        value = -np.sum(np.multiply(np.log(prob+delta) , y))/len(y)
        
        #### write your code above ####
        return value

    def gradient(self, trainX: np.ndarray, trainY: np.ndarray) -> np.ndarray:
        '''
        Compute gradient of logistic regression.
        Parameters:
          trainX is the training data input. trainX is a two two dimensional numpy array.
          trainY is the training data label. trainY is a two dimensional numpy array.
        Return:
          a one dimensional numpy array representing the gradient
        '''
        x = self.artificial_feature(trainX)
        #### write your code below ####

        result=(self.predict(trainX)-trainY)
        g=np.matmul(x.T,result)

        #### write your code above ####
        return g

    def update_weight(self, dLdW: np.ndarray) -> None:
        self.W += -self.learning_rate*dLdW
        
    def one_epoch(self, trainX: np.ndarray, trainY: np.ndarray, batch_size: int, train: bool = True)-> tuple:
        num_sample = trainX.shape[0]
        num_batch = int(num_sample/batch_size)
        batch_index = list(gen_batches(num_sample, num_batch))
        loss_value = 0
        num_correct = 0
        for i, index in enumerate(batch_index):
            X, y = trainX[index,:], trainY[index]
            if train:
                dLdW = self.gradient(X, y)
                self.update_weight(dLdW)
            prob = self.predict(X)
            loss_value += self.loss(y, prob)*X.shape[0]
            num_correct += self.accuracy(y, prob)*X.shape[0]
        loss_value = loss_value/num_sample
        acc = num_correct/num_sample
        return loss_value, acc

    def accuracy(self, y: np.ndarray, prob: np.ndarray)-> float:
        '''
        compute accuracy
        Parameters:
          y is the true label. y is a two dimensional array.
          prob is the predicted label probability. prob is a two dimensional array.
        Return:
          acc is the accuracy value
        '''
        #### write your code below ####
        pred_prob_loc=np.argmax(prob,axis=1)
        pred_y_loc = np.argmax(y, axis=1)
        correct_num=(pred_prob_loc==pred_y_loc).sum()
        #pos = np.zeros(y.shape)
        #pos[:,np.argmax(y, axis=1)] = 1
        acc = correct_num/len(y)
        #### write your code above ####
        return  acc


# In[10]:

def train(model, trainX, trainY, epoches, batch_size):
    loss_value, acc = model.one_epoch(trainX, trainY, batch_size, train = False)
    print('Initialization: ', 'loss %.4f  '%loss_value, 'accuracy %.2f'%acc)
    for epoch in range(epoches):
        loss_value, acc_train = model.one_epoch(trainX, trainY, batch_size)
        print('epoch: %d'%(epoch+1), 'loss %.4f  '%loss_value, 'accuracy %.2f'%acc_train)


# In[11]:

model = MNRegression(num_feature, num_class, learning_rate = 0.01)
train(model, trainX, trainY, epoches = 30, batch_size = 256)


# In[ ]:

test_loss, test_acc = model.one_epoch(testX, testY, batch_size = 256, train = False)
print('testing accuracy is %.4f'%test_acc)


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
