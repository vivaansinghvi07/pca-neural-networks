import torch
from torch import nn

class NormalNetwork(nn.Module):
    """
    Input shape: (N, 728)  -- the flattened images
    Output shape: (N, 10)  -- the class probabilities
    """

class PCANetwork(nn.Module):
    """
    Input shape: (N, x)    -- the principal components
    Output shape: (N, 10)  -- class probabilities

    Write this so that the constructor takes in the parameter `x` - 
    the number of features from PCA - and builds layers according to `x`
    """
    

def train_network(
    net: NormalNetwork | PCANetwork,
    train_data: torch.Tensor, 
    *,
    batch_size: int = 32,
    epochs: int = 20
) -> None:
    """ 
    Train the neural network using `train_data`.
    `train_data` will be in the shape that the neural network requires. 
    For example, if training NormalNetwork, `train_data` will be in the 
    shape (N, 728). Batching is up to you.
    """

def test_network(
    net: NormalNetwork | PCANetwork,
    test_data: torch.Tensor, 
) -> dict[str, float]:
    """ 
    Test the neural network using `test_data`
    `test_data` follows the same shape as `train_data` as above.
    Return a dictionary of strings with measurements of network 
    performance, like accuracy, f1 score, precision, etc.
    """
    ...   
