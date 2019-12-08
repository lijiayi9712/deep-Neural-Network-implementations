import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=22, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    #self.params['W4'] = np.random.normal(0, weight_scale, (3, C, filter_size, filter_size))
    #self.params['b4'] = np.zeros(3)
    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters*H*W/4, hidden_dim))
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b3'] = np.zeros(num_classes)
    
    #self.params['gamma'] = np.ones(num_filters)
    #self.params['beta'] = np.zeros(num_filters)
#     self.params['gamma2'] = np.ones(hidden_dim)
#     self.params['beta2'] = np.zeros(hidden_dim)
#     self.params['gamma3'] = np.ones(hidden_dim)
#     self.params['beta3'] = np.zeros(hidden_dim)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
  
  def conv_bn_relu_pool_backward(dout, cache):
     conv_cache, relu_cache, pool_cache, bn_cache = cache
     ds = max_pool_backward_fast(dout, pool_cache)
     da = relu_backward(ds, relu_cache)
     bn_dx, bn_dg, bn_db = spatial_batchnorm_backward(da, bn_cache)
     dx, dw, db = conv_backward_fast(bn_dx, conv_cache)
     return dx, dw, db, bn_dg, bn_db

  
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    #W4, b4 = self.params['W4'], self.params['b4']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    bn_param = {'mode':'test' if y is None else 'train',}
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #gamma, beta = self.params['gamma'], self.params['beta']
    #cf_out, cf_cache = conv_forward_fast(X, W4, b4, conv_param)
    #print(W1.shape)
    #print(b1.shape)
    #print(X.shape)
    #print(cf_out.shape)
    #cf_out = cf_out[:, 0:2, :, :]
    out, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
 
    ar_out, ar_cache = affine_relu_forward(out, W2, b2)
    scores, cache = affine_forward(ar_out, W3, b3)
     
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    regulation_loss = 0.5*self.reg*np.sum(self.params['W1']**2)+0.5*self.reg*np.sum(self.params['W2']**2)+0.5*self.reg*np.sum(self.params['W3']**2)
    #0.5*self.reg*np.sum(self.params['W4']**2)
    loss += regulation_loss
    ab_dx, ab_dw, ab_db = affine_backward(dscores, cache)
    grads['b3'] = ab_db
    grads['W3'] = ab_dw + self.reg*self.params['W3']
    arb_dx, arb_dw, arb_db = affine_relu_backward(ab_dx, ar_cache)
    grads['b2'] = arb_db
    grads['W2'] = arb_dw + self.reg*self.params['W2']
    dx, dw, db = conv_relu_pool_backward(arb_dx, cache1)
    grads['b1'] = db
    grads['W1'] = dw + self.reg*self.params['W1']
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
