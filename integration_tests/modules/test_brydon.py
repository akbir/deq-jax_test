import jax.numpy as jnp
import jax
import numpy as np
import torch
import pytest
from deq_torch.brydon import th_broyden, th_rmatvec, th_matvec
from deq_jax.src.modules.broyden import broyden, rmatvec, matvec

RNG = np.random.default_rng(42)

def quadratic(x):
    return x ** 2 + x - 5

def sinsodial(x):
    if type(x) == torch.Tensor:
        return np.sin(x)
    return jnp.sin(x)

def linear(x):
    w, b = 2, 1
    return w * x + b

class TestBroydenIntegration:
    @pytest.mark.parametrize('fun', [quadratic, sinsodial, linear])
    def test_optima(self, fun):
        values = RNG.random((2, 3, 4))
        threshold = 30
        eps = 1e-6 * np.sqrt(values.size)
        jax_broyden = jax.jit(broyden, static_argnums=(0, 2, 3))

        jax_ans = jax_broyden(jax.jit(fun), jnp.asarray(values, dtype=jnp.float32), threshold, eps)
        th_ans = th_broyden(fun, torch.tensor(values, dtype=torch.float32), threshold, eps)

        # check they could converge
        assert jax_ans['prot_break'] == th_ans['prot_break']

        # check models are equal
        np.testing.assert_array_almost_equal(jax_ans['trace'][:jax_ans['n_step'] + 1], th_ans['trace'], decimal=4)
        np.testing.assert_array_almost_equal(np.array(jax_ans['result']),
                                             th_ans['result'].numpy(),
                                             decimal=4)

        assert jax_ans['n_step'] == th_ans['nstep']
        assert jax_ans['eps'] == th_ans['eps']

        np.testing.assert_array_almost_equal(jax_ans['diff'], th_ans['diff'], decimal=4)
        np.testing.assert_array_almost_equal(np.array(jax_ans['diff_detail']), th_ans['diff_detail'].numpy(), decimal=4)

    def test_rmatvec(self):
        for _ in range(10):
            values = np.random.rand(2, 3, 4)
            th = torch.tensor(values, dtype=torch.float32)
            ja = jnp.asarray(values, dtype=jnp.float32)

            values_2 = np.random.rand(2, 3, 4, 5)
            values_3 = np.random.rand(2, 5, 3, 4)
            th_2, th_3 = torch.tensor(values_2, dtype=torch.float32), torch.tensor(values_3, dtype=torch.float32)
            ja_2, ja_3 = jnp.asarray(values_2, dtype=jnp.float32), jnp.asarray(values_3, dtype=jnp.float32)

            np.testing.assert_almost_equal(np.array(rmatvec(ja_2, ja_3, ja)),
                                           th_rmatvec(th_2, th_3, th).numpy(), decimal=5)

    def test_matvec(self):
        for _ in range(10):
            values = RNG.random((2, 3, 4))
            th = torch.tensor(values, dtype=torch.float32)
            ja = jnp.asarray(values, dtype=jnp.float32)

            values_2 = RNG.random((2, 3, 4, 5))
            values_3 = RNG.random((2, 5, 3, 4))
            th_2, th_3 = torch.tensor(values_2, dtype=torch.float32), torch.tensor(values_3, dtype=torch.float32)
            ja_2, ja_3 = jnp.asarray(values_2, dtype=jnp.float32), jnp.asarray(values_3, dtype=jnp.float32)

            np.testing.assert_almost_equal(np.array(matvec(ja_2, ja_3, ja)),
                                           th_matvec(th_2, th_3, th).numpy(), decimal=5)
