import jax.numpy as jnp
from cuda import cudart
import os


def get_used_memory():
    status, free, maxm = cudart.cudaMemGetInfo()
    used_memory = maxm - free
    return used_memory / 1024 / 1024 / 1024  #GB


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
initial_used = get_used_memory()
N = 10**8
a = jnp.zeros(N)

print(get_used_memory() - initial_used, "GB")

b = jnp.fft.rfft(a)

print(get_used_memory() - initial_used, "GB")

c = jnp.fft.irfft(b)

print(get_used_memory() - initial_used, "GB")
