import numpy as np 
import torch
import torchvision
from torch.utils.data import DataLoader


def get_mnist_pca(train: bool, count: int):
    """ 
    Returns the MNIST dataset in the form of a tuple, where the 
    first entry is a tensor with shape (n, x) (where x is determined
    by doing PCA) and the second entry is the labels.
    """
    pass

def get_mnist_raw(train: bool, count: int) -> torch.Tensor:
    """ 
    Returns the MNIST dataset in the form of a tuple, where the 
    first entry is a normalized tensor with shape (n, 28 * 28) and 
    the second entry is the labels. 
    """
    ...
    
