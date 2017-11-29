#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:45:16 2017

@author: yafei
"""
from torch import nn

def get_activation_by_name(name):
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "sigmoid":
        return nn.Sigmoid()
    elif name.lower() == "tanh":
        return nn.Tanh()
    elif name.lower() == "softmax":
        return nn.Softmax()
    elif name.lower() == "none" or name.lower() == "linear":
        return nn.Linear()
    else:
        raise Exception(
            "unknown activation type: {}".format(name)
          )
        
