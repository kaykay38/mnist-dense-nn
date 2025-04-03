# Author: Mia Hunt

# Various math functions, including a collection of activation functions used in NN.

import numpy as np

class MyMath:

    def tanh(x):
        ''' tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh of the corresponding element in array x
        '''
        return np.tanh(x)


    def tanh_de(x):
    
        ''' Derivative of the tanh function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is tanh derivative of the corresponding element in array x
        '''
        ## d/dx tanh(x) = sech^2(x)
        ## sech^2(x) = 1/cosh^2(x) = 2/(cosh(2x)+1)
        # return 2/(np.cosh(2*x) + 1)
        tanh_de_vec = np.vectorize(lambda x: 1/pow(np.cosh(x),2))
        return tanh_de_vec(x)
    
    def logis(x):
        ''' Logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic of 
                    the corresponding element in array x
        '''
        ## f(x) = 1/(1+e^(-x)) 
        sig_vec = np.vectorize(lambda x: 1/(1+np.exp(-x)))
        return sig_vec(x)

    
    def logis_de(x):
        ''' Derivative of the logistic function. 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is logistic derivative of 
                    the corresponding element in array x
        '''
        ## d/dx logis(x) = logis(x) * (1 - logis(x))
        ## https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
        return MyMath.logis(x) * (1 - MyMath.logis(x))

    
    def iden(x):
        ''' Identity function
            Support vectorized operation
            
            x: an array type of real numbers
            return: the numpy array where every element is the same as
                    the corresponding element in array x
        '''
        return np.array(x)

    
    def iden_de(x):
        ''' The derivative of the identity function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array of all ones of the same shape of x.
        '''
        x = np.array(x)
        return np.ones(x.shape)
        

    def relu(x):
        ''' The ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the max of: zero vs. the corresponding element in x.
        '''
        relu_vec = np.vectorize(lambda x: np.maximum(0.0,x))
        return relu_vec(x)

    
    def _relu_de_scaler(x):
        ''' The derivative of the ReLU function. Scaler version.
        
            x: a real number
            return: 1, if x > 0; 0, otherwise.
        '''
        ## f(x) = 1/(1+e^(-x)) 
        if x > 0:
            return 1
        return 0

    
    def relu_de(x):
        ''' The derivative of the ReLU function 
            Support vectorized operation

            x: an array type of real numbers
            return: the numpy array where every element is the _relu_de_scaler of the corresponding element in x.   
        '''
        relu_de_vec = np.vectorize(lambda x: MyMath._relu_de_scaler(x))
        return relu_de_vec(x)
