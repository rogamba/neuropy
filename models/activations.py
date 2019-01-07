import numpy as np
import math

''' Misc of activation functions 
    for the neural network
'''


def get_activation(name):
    return eval(name+"_activation")

def sigmoid_activation(x=None,deriv=False):
    ''' Sigmoid function (used for hidden neurons)
    '''
    if deriv == True:
        return sigmoid(x)*(1-sigmoid(x))
    return 1/(1+np.exp(-x))

def tansig_activation(x=None,deriv=False):
    ''' Tangent Sigmoid function
    '''
    return ( np.exp(x)-np.exp(-x) ) / ( np.exp(x)+np.exp(-x) )

def purelin_activation(x=None,deriv=False):
    ''' Linear (usually used in output neurons)
    '''
    if deriv == True:
        return 1 
    return x 

def tanh_activation(z):
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)

def sin_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.sin(z)

def gauss_activation(z):
    z = max(-3.4, min(3.4, z))
    return math.exp(-5.0 * z**2)

def relu_activation(z):
    return z if z > 0.0 else 0.0

def softplus_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 0.2 * math.log(1 + math.exp(z))

def identityn_activation(z):
    return z

def clamped_activation(z):
    return max(-1.0, min(1.0, z))

def inverse_activation(z):
    if z == 0:
        return 0.0

    return 1.0 / z

def logarithmic_activation(z):
    z = max(1e-7, z)
    return math.log(z)

def exponential_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.exp(z)

def absolute_activation(z):
    return abs(z)

def hat_activation(z):
    return max(0.0, 1 - abs(z))

def square_activation(z):
    return z ** 2

def cube_activation(z):
    return z ** 3
