import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import distance


class Loss_orthogonal(nn.Module):
    def __init__(self):
        super(Loss_orthogonal, self).__init__()

    def forward(self, x1, x2):
        assert x1.shape == x2.shape
        error = x1 @ x2.transpose(-2, -1)
        ort = torch.mean(error.view(-1))
        return ort

class Loss_distance(nn.Module):
    def __init__(self):
        super(Loss_distance, self).__init__()

    def forward(self, x1, x2, method):
        assert x1.shape == x2.shape
        if method =='euclidean':
            error = torch.pow(x1-x2, 2)
            euc = torch.sum(error, dim=1)
            dis = torch.sqrt(torch.mean(euc.contiguous().view(-1)))
        if method == 'cosine':
            error = torch.cosine_similarity(x1, x2)
            cos = torch.mean(error.view(-1))
            dis = 1 - cos
        return dis
