# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#对Variable操作如果直接使用索引，梯度不传播
import torch
import torch.nn as nn
import torch.nn.functional as F
def one_hot(index,classes):
    N=index.size(0)
    onehot = torch.zeros(N, classes).long().cuda()
    onehot_convert=onehot.scatter_(dim=1, index=torch.unsqueeze(index, dim=1), src=torch.ones(N, classes).long().cuda())
    return  onehot_convert.float()
class ratio(nn.Module):
    def __init__(self, eps=1e-7,alpha=1,beta=0.25):
        super(ratio, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        return
    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        Psoft = torch.nn.functional.softmax(input,dim=1)
        Psoft = Psoft.clamp(1e-7, 1.-1e-7)
        Loss1=0.0
        Loss1+=torch.sum(torch.log((((self.alpha-torch.sum(y*Psoft,1))**self.beta)/(torch.sum(y*Psoft,1))+1e-7)))
        Loss1=Loss1/(target.size(0))
        return Loss1
        
        


        


        
