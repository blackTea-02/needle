"""Operator implementations."""

from typing import Optional

from needle.autograd import NDArray
from needle.autograd import Tensor, TensorOp
from needle import init

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * self.scalar,


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad: Tensor, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        b = node.inputs[1]
        p_a = multiply(power(a, add_scalar(b, -1)), b) # 输出对a的偏导：b * a^(b - 1)
        p_b = multiply(power(a, b), log(a)) # 输出对b的偏导：a^b * lna
        return  out_grad * p_a, out_grad * p_b
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # scalar为int，将其转为一个常量Tensor，放在和a相同的设备上，但不参与计算图计算
        scalar = Tensor(self.scalar, device=a.device).detach()
        # 将scalar的纬度广播至和a相同
        scalar = broadcast_to(scalar, a.shape)
        d_a = multiply(power(a, add_scalar(scalar, -1)), scalar)  # 输出对a的导数：scalar * a^(scalar - 1)
        return out_grad * d_a
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        b = node.inputs[1]
        # 定义“1”的Tensor并广播至和b相同
        one = init.ones(*b.shape, device=b.device).detach()
        p_a = divide(one, b) # 输出对a的偏导为：1 / b
        p_b = divide(negate(a), power_scalar(b, 2)) # 输出对b的偏导为：(-a) / b^2
        return out_grad * p_a, out_grad * p_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # 定义“1”的Tensor并广播至和a相同
        one = init.ones(*a.shape, device=a.device).detach()
        # scalar为int，将其转为一个常量Tensor，将scalar的纬度广播至和a相同
        scalar = Tensor(self.scalar, device=a.device).detach()
        scalar = broadcast_to(scalar, a.shape)
        d_a = divide(one, scalar)
        return out_grad * d_a
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    # axes为需要交换的两个轴，使用numpy作为后端时，参数需要传入所有的轴
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes_t = list(x for x in range(a.ndim))
        if self.axes is not None:
            axes_t[self.axes[0]] = self.axes[1]
            axes_t[self.axes[1]] = self.axes[0]
        else:
            axes_t[-2], axes_t[-1] = axes_t[-1], axes_t[-2]
        self.axes_t = tuple(axes_t)
        return array_api.transpose(a, axes=self.axes_t)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # broadcast_to算子的梯度就是将广播的纬度进行reduce_sum
        a = node.inputs[0]
        # a和out_grad的形状
        a_shape = a.shape
        out_grad_shape = out_grad.shape
        # 将a的纬度数量扩展至out_grad的纬度数量，扩展出的新维度长度为1
        len_a = len(a_shape)
        len_out_grid = len(out_grad_shape)
        a_shape_expand = [1] * len_out_grid
        a_shape_expand[len_out_grid - len_a:] = a_shape
        # 找到广播的纬度
        axis = []
        for index, value in enumerate(a_shape_expand):
            if a_shape_expand[index] == 1 and \
                    out_grad_shape[index] != 1:
                axis.append(index)
        axis = tuple(axis)
        # 在对应纬度进行reduce
        out_grad = summation(out_grad, axis)
        # 将reduce后的梯度reshape为a的形状
        return reshape(out_grad, a_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            axes = tuple([axes])
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # sum的梯度就是将out_grad广播为与输入相同的形状
        a = node.inputs[0]
        # 先将out_grad进行reshape，使其维度数量与输入相同，且被sum的纬度长度为1
        out_grad_shape_expand = [1] * len(a.shape)
        # 将b的在某个维度的长度写入b_shape_expand，如果该维度为sum进行的纬度，则写入1
        # offset的表示已经将多少个被sum的纬度写入了1
        # 如果axes为None则跳过for循环
        if self.axes is not None:
            offset = 0
            for index, _ in enumerate(out_grad_shape_expand):
                if offset < len(self.axes) and self.axes[offset] == index:
                    offset += 1
                else:
                    out_grad_shape_expand[index] = out_grad.shape[index - offset]
        out_grad = reshape(out_grad, out_grad_shape_expand)
        return broadcast_to(out_grad, a.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        a_t = transpose(a)
        b_t = transpose(b)
        d_a = matmul(out_grad, b_t)
        d_b = matmul(a_t, out_grad)
        # 考虑多个批次的情况
        # 对于X(B, M, K) matmul W(K, N)情况，实际上W(K, N)被广播到了W(B, K, N)，计算出的W梯度也为(B, K, N)，
        # 所以W的梯度算完后需要reduce为(K, N)
        # 而X(B, M, K)的梯度不需要求和，因为反向传播时，out_grid的形状包含B纬度，X的梯度形状天然与X形状相同
        if len(d_a.shape) > len(a.shape):
            axes = tuple(x for x in range(len(d_a.shape) - len(a.shape)))
            d_a = summation(d_a, axes)
        if len(d_b.shape) > len(b.shape):
            axes = tuple(x for x in range(len(d_b.shape) - len(b.shape)))
            d_b = summation(d_b, axes)
        return d_a, d_b

        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        negate_one = negate(init.ones(*a.shape, device=a.device)).detach()
        return out_grad * negate_one
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        one = init.ones(*a.shape, device=a.device).detach()
        d_a = divide(one, a)
        return out_grad * d_a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # 生成对应的bool掩码矩阵
        mask = array_api.greater(a, 0)
        return array_api.multiply(a, mask)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        mask = Tensor(array_api.greater(a, 0).astype('float32'))
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

