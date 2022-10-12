import jax.numpy as jnp
from cuda import cudart
import os
from jax import grad, vmap


def get_used_memory():
    status, free, maxm = cudart.cudaMemGetInfo()
    used_memory = maxm - free
    return used_memory / 1024 / 1024 / 1024  #GB


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
initial_used = get_used_memory()
N = 10**8


def f(x):
    return jnp.sin(x)


df = vmap(grad(f))
a = jnp.zeros(N)
print("estimate for array allocation:", N * 4 / (1024**3), "GB")
print("array allocation: ", get_used_memory() - initial_used, "GB")

b = df(a)
print("after grad: ", get_used_memory() - initial_used, "GB")

c = jnp.fft.rfft(a)

print("after RFFT: ", get_used_memory() - initial_used, "GB")

d = jnp.fft.irfft(c)

print("after IRFFT: ", get_used_memory() - initial_used, "GB")
