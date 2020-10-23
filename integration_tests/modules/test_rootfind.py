import copy

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
from jax import value_and_grad, grad

from deq_jax.src.modules.rootfind import deq
from deq_torch.rootfind import DEQModule, RootFind


class Pointwise(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = torch.nn.Parameter(weight * torch.ones(1, 2, 3), requires_grad=True)
        self.register_parameter(name='W', param=self.weight)

        self.bias = torch.nn.Parameter(bias * torch.ones(1, 2, 3), requires_grad=True)
        self.register_parameter(name='B', param=self.bias)

    def forward(self, z1ss, *args):
        return self.weight * z1ss + self.bias

    def copy(self, func):
        self.weight.data = func.weight.data.clone()
        self.bias.data = func.bias.clone()

class TLinear(nn.Module):
    def __init__(self, output_size, weight, bias):
        super().__init__()
        self.linear = nn.Linear(output_size, output_size, bias=True)
        with torch.no_grad():
            self.linear.weight.fill_(weight)
            self.linear.bias.fill_(bias)

    def forward(self, z1ss, *args):
        return self.linear(z1ss)

    def copy(self, func):
        self.linear.weight.data = func.linear.weight.clone()
        self.linear.bias.data = func.linear.bias.clone()


class LinearDEQ(DEQModule):
    def __init__(self, func, func_copy):
        super(LinearDEQ, self).__init__(func, func_copy)

    def forward(self, z1s):
        zero = torch.zeros_like(z1s)
        threshold, train_step = 30, -1
        z1s_out = RootFind.apply(self.func,
                                 z1s,
                                 None,
                                 None,
                                 None,
                                 threshold,
                                 train_step)

        z1s_out = RootFind.f(self.func, z1s_out, zero, zero, zero, threshold, train_step)
        z1s_out = self.Backward.apply(self.func_copy, z1s_out, zero, zero, zero, threshold, train_step)
        return z1s_out


def test_parity_with_jax():
    weight = -3
    bias = 10

    def layer(params, x):
        return params['w'] * x + params['b']

    def toy_model(params, data):
        z = deq(layer, 30, params, data, True)
        return jnp.sum(z)

    params = {'w': weight * jnp.ones((1, 2, 3)), 'b': bias * jnp.ones((1, 2, 3))}
    data = jnp.ones((1, 2, 3))

    value, grad_params = value_and_grad(toy_model)(params, data)
    grad_input = grad(toy_model, argnums=1)(params, data)

    class DEQ(nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            self.func = Pointwise(weight, bias)
            self.func_copy = copy.deepcopy(self.func)
            for params in self.func_copy.parameters():
                params.requires_grad_(False)

            self.deq = LinearDEQ(self.func, self.func_copy)

        def forward(self, input):
            self.func_copy.copy(self.func)
            return self.deq(input)

    th_net = DEQ(weight, bias)
    input = torch.ones(1, 2, 3, requires_grad=True)

    torch_value = th_net(input).sum()
    torch_value.backward()
    torch_w_gradient = th_net.func.W.grad
    torch_b_gradient = th_net.func.B.grad

    np.testing.assert_almost_equal(value, torch_value.detach().numpy())
    np.testing.assert_almost_equal(grad_input, input.grad.detach().numpy())
    np.testing.assert_almost_equal(np.asarray(grad_params['b']), torch_b_gradient.detach().numpy())
    np.testing.assert_almost_equal(np.asarray(grad_params['w']), torch_w_gradient.detach().numpy())


def test_parity_with_haiku():
    weight, bias = 1, 5

    def build_forward(output_size, max_iter):
        def forward_fn(x: jnp.ndarray) -> jnp.ndarray:
            linear_1 = hk.Linear(output_size,
                                 name='l1',
                                 w_init=hk.initializers.Constant(weight),
                                 b_init=hk.initializers.Constant(bias))

            transformed_linear = hk.without_apply_rng(
                hk.transform(linear_1)
            )
            inner_params = hk.experimental.lift(
                transformed_linear.init)(hk.next_rng_key(), x)

            return deq(transformed_linear.apply, max_iter, inner_params, x, True)
        return forward_fn

    input = 2*jnp.ones((1, 2, 3))
    rng = jax.random.PRNGKey(42)
    forward_fn = build_forward(3, 30)
    forward_fn = hk.transform(forward_fn)
    params = forward_fn.init(rng, input)

    # @jax.jit
    def loss_fn(params, rng, x): return jnp.sum(forward_fn.apply(params, rng, x))

    value = loss_fn(params, rng, input)
    grad_params = grad(loss_fn)(params, rng, input)

    class DEQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.func = TLinear(3, weight, bias)
            self.func_copy = copy.deepcopy(self.func)
            for params in self.func_copy.parameters():
                params.requires_grad_(False)
            self.deq = LinearDEQ(self.func, self.func_copy)

        def forward(self, input):
            self.func_copy.copy(self.func)
            # return self.deq(input)
            return self.deq(input)

    th_net = DEQ()
    input = 2 * torch.ones(1, 2, 3, requires_grad=True)

    torch_value = th_net(input).sum()
    torch_value.backward()
    torch_w_gradient = th_net.func.linear.weight.grad
    torch_b_gradient = th_net.func.linear.bias.grad

    np.testing.assert_almost_equal(value, torch_value.detach().numpy())
    np.testing.assert_almost_equal(np.asarray(grad_params['lifted/l1']['b']), torch_b_gradient.detach().numpy())
    np.testing.assert_almost_equal(np.asarray(grad_params['lifted/l1']['w']), torch_w_gradient.detach().numpy())
