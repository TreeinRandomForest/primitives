import torch
import torch.nn as nn
import torch_geometric.nn as gnn

'''This is a first attempt to classify and organize primitive
operations for deep learning.

We don't care about correct initialization of weights for anything
below.

Notes:
- The tensor ops are implemented in ATen (https://pytorch.org/cppdocs/).
- Dropping classes for now - just vanilla listing of functions and tests. Can organize later
'''

#----------identity-------------
def id(x):
    return x

def test_id():
    x = torch.randn(3, 4)
    y = id(x)

    assert (x==y).flatten().sum() == torch.prod(torch.tensor(x.shape))

#----------linear-------------
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

def test_linear_ops():
    X = torch.randn((B, in_f))
    val = linear_ops(X, W, b)

    assert val.shape == (B, out_f)

def bilinear_ops(X1, X2, W, b):
    pass
    
class BiLinear:
    '''
    Matrix element of W + bias i.e.
    x1^T W x2 + b
    '''

    #tensor shapes
    B = 5 #batch size
    
    in_shape = 4
    out_shape = 4

    
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
