"""The module.
"""
from typing import List

import needle.init as init
from needle import ops
from needle.autograd import Tensor


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
            self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features,
                                                     device=device,
                                                     dtype=dtype,
                                                     require_grad=True
                                                     ))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1,
                                                       device=device,
                                                       dtype=dtype,
                                                       require_grad=True
                                                       )).reshape((1, out_features))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X_matmul = X @ self.weight
        if self.bias:
            # ndl不支持隐式广播，必须显示广播
            return X_matmul + ops.broadcast_to(self.bias, X_matmul.shape)
        return X_matmul
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # 计算一个样本长度
        length = 1
        for i, dim in enumerate(X.shape):
            if i == 0:
                continue
            length *= dim
        return X.reshape((X.shape[0], length))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # softmax-loss：对预测向量做LSE，减去预测正确的概率，求和，除以batch
        exp_sum = ops.logsumexp(logits, axes=(1,))
        num_batch = logits.shape[0]
        num_class = logits.shape[1]
        y_one_hot = init.one_hot(num_class, y, device=logits.device)
        # 计算实际概率
        real_possibility = (logits * y_one_hot).sum(axes=(1,))
        return (exp_sum - real_possibility).sum() / num_batch
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, dim,
                                          device=device,
                                          dtype=dtype,
                                          requires_grad=True
                                          ))
        self.bias = Parameter(init.zeros(1, dim,
                                         device=device,
                                         dtype=dtype,
                                         requires_grad=True
                                         ))
        self.running_mean = init.zeros(dim,
                                       device=device,
                                       dtype=dtype,
                                       requires_grad=True
                                       )
        self.running_var = init.ones(dim,
                                     device=device,
                                     dtype=dtype,
                                     requires_grad=True
                                     )

        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        feature = x.shape[1]
        if self.training:
            # 计算mean和var并广播为与x相同的形状
            mean = x.sum(axes=(0,)) / batch_size
            mean_broadcast = ops.broadcast_to(mean, x.shape)
            var = ops.power_scalar(x - mean_broadcast, 2).sum(axes=(0,)) / batch_size
            var_broadcast = ops.broadcast_to(var, x.shape)
            # 更新全局mean和var
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            # 计算结果
            out = (x - mean_broadcast) / ops.power_scalar(var_broadcast + self.eps, 0.5)
            weight_broadcast = ops.broadcast_to(self.weight, x.shape)
            bias_broadcast = ops.broadcast_to(self.bias, x.shape)
            return weight_broadcast * out + bias_broadcast
        else:
            mean_broadcast = ops.broadcast_to(self.running_mean, x.shape)
            var_broadcast = ops.broadcast_to(self.running_var, x.shape)
            out = (x - mean_broadcast) / ops.power_scalar(var_broadcast + self.eps, 0.5)
            weight_broadcast = ops.broadcast_to(self.weight, x.shape)
            bias_broadcast = ops.broadcast_to(self.bias, x.shape)
            return weight_broadcast * out + bias_broadcast

        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # weight和bias均为一维向量
        self.weight = Parameter(init.ones(dim,
                                          device=device,
                                          dtype=dtype,
                                          requires_grad=True
                                          ))
        self.bias = Parameter(init.zeros(dim,
                                         device=device,
                                         dtype=dtype,
                                         requires_grad=True
                                         ))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        feature = x.shape[1]
        # 计算mean后，将mean转置，并广播为x的形状
        mean_reshape_broadcast = ops.broadcast_to((x.sum(axes=(1,)) / feature).reshape((batch_size, 1)), x.shape)
        var = ops.power_scalar(x - mean_reshape_broadcast, 2).sum(axes=(1,)) / feature
        # 计算var后，将var转置，并广播为x的形状
        var_reshape_broadcast = ops.broadcast_to(var.reshape((batch_size, 1)), x.shape)
        out = (x - mean_reshape_broadcast) / ops.power_scalar((var_reshape_broadcast + self.eps), 0.5)
        # 广播weight和bias
        weight_broadcast = ops.broadcast_to(self.weight, x.shape)
        bias_broadcast = ops.broadcast_to(self.bias, x.shape)
        return weight_broadcast * out + bias_broadcast
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            batch_size = x.shape[0]
            feature = x.shape[1]
            mask = init.randb(batch_size, feature,
                              p=self.p,
                              device=x.device,
                              requires_grad=False
                              )  / (1 - self.p)
            return x * mask
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
