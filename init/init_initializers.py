import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a,
                device=kwargs.get('device', None),
                dtype=kwargs.get('dtype', 'float32'),
                requires_grad=kwargs.get('requires_grad', False)
                )
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # 注意：hw2中的题目所给公式std不需要平方
    std = gain *  math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0.0, std=std,
                 device=kwargs.get('device', None),
                 dtype=kwargs.get('dtype', 'float32'),
                 requires_grad=kwargs.get('requires_grad', False)
                 )
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    a = math.sqrt(2) * math.sqrt(3 / fan_in)
    return rand(fan_in, fan_out, low=-a, high=a,
                device=kwargs.get('device', None),
                dtype=kwargs.get('dtype', 'float32'),
                requires_grad=kwargs.get('requires_grad', False)
                )
    ### END YOUR SOLUTION



def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # 注意：hw2中的题目所给公式std不需要平方
    std = math.sqrt(2) / math.sqrt(fan_in)
    return randn(fan_in, fan_out, mean=0.0, std=std,
                 device=kwargs.get('device', None),
                 dtype=kwargs.get('dtype', 'float32'),
                 requires_grad=kwargs.get('requires_grad', False)
                 )
    ### END YOUR SOLUTION