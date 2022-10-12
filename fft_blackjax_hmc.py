import sys
from tkinter import Y
import jax.numpy as jnp
from cuda import cudart
import os
from jax import grad, vmap
from jax.config import config
import jax
#config.update('jax_enable_x64', True)


def get_used_memory():
    status, free, maxm = cudart.cudaMemGetInfo()
    used_memory = maxm - free
    return used_memory / 1024 / 1024 / 1024  #GB


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
initial_used = get_used_memory()
N = int(1.3*10**8)


def f(x, c):
    return jnp.sin(x + c)


a = jnp.linspace(0.0, 2.0 * jnp.pi, N)
print("estimate for array allocation:", N * 4 / (1024**3), "GB")
print("array allocation: ", get_used_memory() - initial_used, "GB")


def g(c):
    return jnp.real(jnp.fft.rfft(jnp.real(f(a, c))))


d = g(0.1)
print("after computing g: ", get_used_memory() - initial_used, "GB")

import jax.scipy.stats as stats
import blackjax

print("after importing blackjax/numpyro: ",
      get_used_memory() - initial_used, "GB")


def logprob_fn(x):
    logpdf = stats.norm.logpdf(d, g(x["loc"]), 0.01)
    return jnp.sum(logpdf)


step_size = 1e-3
inverse_mass_matrix = jnp.array([1., 1.])
nuts = blackjax.nuts(logprob_fn, step_size, inverse_mass_matrix)
#nuts = blackjax.hmc(logprob_fn, step_size, inverse_mass_matrix, 0.01)

# Initialize the state
initial_position = {
    "loc": 0.15,
}
state = nuts.init(initial_position)
print("BlackJAX initialization : ", get_used_memory() - initial_used, "GB")

# Iterate
rng_key = jax.random.PRNGKey(0)
for _ in range(100):
    _, rng_key = jax.random.split(rng_key)
    y = nuts.step(rng_key, state)
    print(y)
    state, _ = y

