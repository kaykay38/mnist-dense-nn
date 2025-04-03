# Author: Mia Hunt

# Implementation of the forwardfeed neural network using stochastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forwardfeed/backtracking round trip. 

from tqdm import tqdm
import numpy as np
import nn.math_util as mu
import nn.nn_layer as nn_layer

class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L. 
    
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        self.layers.append(nn_layer.NeuralLayer(d, act)) 
        self.L = self.L + 1
    

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''
        layer = self.layers
        for l in range(1, self.L+1):
            layer[l].W = np.random.uniform(-1/np.sqrt(layer[l].d), 1/np.sqrt(layer[l].d), (layer[l-1].d + 1, layer[l].d))
            
    
        
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 20):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.


        ## prep the data: add bias column; randomly shuffle data training set. 
        X = np.insert(X,0,1,axis=1)
        layer = self.layers

        error = 0
        n = X.shape[0]
        if SGD:
            mini_batches = NeuralNetwork._get_mini_batches(X, Y, mini_batch_size)
            i = 0

        ## for every iteration:
        for _ in tqdm(range(iterations)):

            # get a minibatch and use it for SGD
            if SGD:
                if i > len(mini_batches) - 1:
                    i = 0
                layer[0].X = mini_batches[i][0]
                Y = mini_batches[i][1]
                n = layer[0].X.shape[0]
                i += 1
            else:
                layer[0].X = X
            
            # forwardfeeding
            for l in range(1, self.L + 1):
                layer[l].S =  layer[l-1].X @ layer[l].W
                layer[l].X = np.insert(layer[l].act(layer[l].S),0,1,axis=1)

            # calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
            error = np.sum((layer[self.L].X[:,1:] - Y) * (layer[self.L].X[:,1:] - Y)) / n

            layer[self.L].Delta = 2 * (layer[self.L].X[:,1:] - Y) * layer[self.L].act_de (layer[self.L].S)
            layer[self.L].G = np.einsum('ij,ik -> jk', layer[self.L-1].X, layer[self.L].Delta) / n

            # backpropagation to calculate the gradients of all the weights
            for l in range(self.L - 1, 0, -1):
                layer[l].Delta = layer[l].act_de(layer[l].S) * (layer[l+1].Delta @ layer[l+1].W[1:].T)
                layer[l].G = np.einsum('ij,ik -> jk', layer[l-1].X, layer[l].Delta) / n
            
            # use the gradients to update all the weight matrices. 
            for l in range(1, self.L + 1):
                layer[l].W = layer[l].W - eta * layer[l].G



    # function to return a list containing mini-batches
    def _get_mini_batches(X, y, batch_size):
        mini_batches = []
        data = np.concatenate((X,y), axis=1) # [[x_features,y_labels],[x_features,y_labels],...]
        d = X.shape[1]
        
        np.random.shuffle(data)
        n_batches = data.shape[0] // batch_size
        if data.shape[0] % batch_size != 0:
            n_batches += 1
        
        for i in range(0, n_batches):
            X_mini = data[i * batch_size : (i + 1) * batch_size, : d]
            y_mini = data[i * batch_size : (i + 1) * batch_size, d :]
            mini_batches.append((X_mini,y_mini))
            
        return mini_batches



    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.

                    s = the likelihood (not probability  b/c not normalized) of each class id column

                    y = take the largest class id likelihood *index* put into corresponding y row.
         '''
        layer = self.layers
        layer[0].X = np.insert(X,0,1,axis=1)
        for l in range(1, self.L + 1):
            layer[l].S = layer[l-1].X @ layer[l].W
            layer[l].X = layer[l].act(np.insert(layer[l].S,0,1,axis=1))
        y_pred = np.argmax(layer[self.L].X[:,1:],axis=1).reshape(-1,1)
        return y_pred
   

    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        Y_result = np.argwhere(Y==1)[:,1].reshape(-1,1)
        y_pred = self.predict(X)
        misclassfied_num = np.sum(y_pred != Y_result)
        return misclassfied_num/Y.shape[0]
