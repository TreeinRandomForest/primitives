import torch
import torch.nn as nn
import torch_geometric.nn as gnn

import sys

'''This is a first attempt to classify and organize primitive
operations for deep learning.

We don't care about correct initialization of weights for anything
below.

Notes:
- The tensor ops are implemented in ATen (https://pytorch.org/cppdocs/).
- Dropping classes for now - just vanilla listing of functions and tests. Can organize later
- Don't correct about correct initializations of weights
- Need to revisit broadcasting rules (e.g. bilinear operates on only last dimension of inputs)
'''

#----------identity-------------
def id(x):
    return x

def test_id():
    x = torch.randn(3, 4)
    y = id(x)

    assert (x==y).flatten().sum() == torch.prod(torch.tensor(x.shape))

#----------linear-------------
# out = W X + b
# since the first dim of X is batch size, this is commonly implemented as:
# out = X W^T + b (broadcasting used for addition)
B = 5 #batch_size
in_f = 10 #number of features
out_f = 7 #number of output activations
    
in_shape = (B, in_f)
out_shape = (B, out_f)

#layer weights
W = torch.rand((out_f, in_f))
b = torch.rand((out_f,))

#promised not to worry about weight init but will add some info
W = (2 * W - 1) * torch.sqrt(torch.tensor(1./in_f))
b = (2 * b - 1) * torch.sqrt(torch.tensor(1./in_f))

#ops
# out = W X + b
# W - (out_f, in_f)
# X - (B, in_f)
# b - (out_f,)
def linear_ops(X, W, b):
    val = torch.add(torch.matmul(X, W.T), b)

    return val
    
def test_linear_ops():
    X = torch.randn((B, in_f))

    m = nn.Linear(in_f, out_f)

    val_custom = linear_ops(X, m.weight, m.bias)
    val_torch = m(X)
    
    assert val_custom.shape == (B, out_f)
    assert val_torch.shape == (B, out_f)
    assert torch.allclose(val_custom, val_torch)
    
test_linear_ops()

#-----------bilinear---------
# Transformation is: x1^T A x2 + b
# This is a matrix element (excluding the bias) i.e. x1^T A x2 is a scalar
# There are several A's which are packed into a 3d tensor
# Pack: x1^T A_i x2 + b_i where 0 <= i < A.shape[0](==b.shape[0])

B = 5 #batch_size
in1_f = 10 #number of features for input 1
in2_f = 15 #number of features for input 2
out_f = 7 #number of output activations
    
in1_shape = (B, in1_f)
in2_shape = (B, in2_f)
out_shape = (B, out_f)

#weights
W = torch.randn((out_f, in1_f, in2_f))
b = torch.randn((out_f,))

#distribution
W = (2*W - 1) * torch.sqrt(torch.tensor(1./in1_f))
b = (2*b - 1) * torch.sqrt(torch.tensor(1./in1_f))

def bilinear_ops(X1, X2, W, b):
    '''There might be more efficient ways of implementing this
    '''
    out = torch.bmm(torch.matmul(X1, W).permute(1,0,2), X2.unsqueeze(2)).squeeze(2) + b

    return out

def test_bilinear_ops():
    X1 = torch.randn((B, in1_f))
    X2 = torch.randn((B, in2_f))

    m = nn.Bilinear(in1_f, in2_f, out_f)

    val_custom = bilinear_ops(X1, X2, m.weight, m.bias)
    val_torch = m(X1, X2)
    
    assert val_custom.shape == (B, out_f)
    assert val_torch.shape == (B, out_f)
    assert torch.allclose(val_custom, val_torch)

test_bilinear_ops()

#-----------lazy linear---------
# Parameters (weights and biases) are unitialized
# by the constructor. Parameters are initialized
# by the first forward call
# https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin

sys.exit(0)

class Convolutions:
    pass

class Pooling:
    pass

class Activations:
    pass

class Normalizations:
    pass

class RecurrentLayers:
    pass

class TransformerLayers:
    pass

class DropoutLayers:
    pass

class GNNLayers:
    #68 (!) layers
    gnn_layers = [l for l in dir(gnn) if l.find('Conv')>-1]
