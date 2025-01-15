"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            # 获取累计梯度，没有累计梯度则生成一个全0矩阵
            u_old = self.u.get(param, ndl.init.zeros(
                *param.shape,
                device=param.device,
                dtype=param.dtype,
                requires_grad=False
            ))
            u_new = self.momentum * u_old.data + (1 - self.momentum) * param.grad.data
            u_new = ndl.Tensor(u_new, device=param.device, dtype=param.dtype, requires_grad=False)
            self.u[param] = u_new
            param.data = param.data - self.lr * (u_new.data + self.weight_decay * param.data)
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            grad = ndl.Tensor(param.grad, device=param.device, dtype=param.dtype, requires_grad=False).data\
                   + param.data * self.weight_decay
            m_old = self.m.get(param, ndl.init.zeros(
                *param.shape,
                device=param.device,
                dtype=param.dtype,
                requires_grad=False
            ))
            v_old = self.v.get(param, ndl.init.zeros(
                *param.shape,
                device=param.device,
                dtype=param.dtype,
                requires_grad=False
            ))
            m_new = self.beta1 * m_old.data + (1 - self.beta1) * grad.data
            v_new = self.beta2 * v_old.data + (1 - self.beta2) * (grad.data ** 2)
            self.m[param] = m_new
            self.v[param] = v_new
            m_hat = m_new / (1 - self.beta1 ** self.t)
            v_hat = v_new / (1 - self.beta2 ** self.t)
            param.data = param.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
        ### END YOUR SOLUTION
