from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        return Z - logsumexp(Z, axes=(1,))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, self.axes, keepdims=True)
        Z_t = array_api.log(
            array_api.sum(array_api.exp(Z - max_z), axis=self.axes, keepdims=True)
        ) + max_z
        # 将被sum的纬度消去
        if self.axes:
            shape = [d for i, d in enumerate(Z.shape) if i not in self.axes]
        else:
            shape = []
        return array_api.reshape(Z_t, tuple(shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # 将梯度的纬度数量扩展到与输入相同的数量，并reshape梯度
        Z = node.inputs[0]
        # if内的原理参见summation的gradient
        shape = [1] * len(Z.shape)
        if self.axes:
            offset = 0
            for index, _ in enumerate(shape):
                if offset < len(self.axes) and self.axes[offset] == index:
                    offset += 1
                else:
                    shape[index] = out_grad.shape[index - offset]
        out_grad = reshape(out_grad, shape)
        node = reshape(node, shape)
        # LogSumExp的导数为输入的softmax
        # softmax和lse之间的相互转换：softmax(x) = exp(x - LSE(x))
        # node即为LogSumExp(x)
        return out_grad * exp(Z + (-node))
    ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

