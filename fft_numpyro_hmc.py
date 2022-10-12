import sys
import jax.numpy as jnp
from cuda import cudart
import os
from jax import grad, vmap
from jax.config import config
#config.update('jax_enable_x64', True)

def get_used_memory():
    status, free, maxm = cudart.cudaMemGetInfo()
    used_memory = maxm - free
    return used_memory / 1024 / 1024 / 1024  #GB


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
initial_used = get_used_memory()
N = int(2*10**8)


def f(x,c):
    return jnp.sin(x+c)

a = jnp.linspace(0.0,2.0*jnp.pi,N)
print("estimate for array allocation:", N * 4 / (1024**3), "GB")
print("array allocation: ", get_used_memory() - initial_used, "GB")

def g(c):
    return jnp.real(jnp.fft.rfft(jnp.real(f(a,c))))

d = g(0.1) 
print("after computing g: ", get_used_memory() - initial_used, "GB")

import numpyro.distributions as dist
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from numpyro.diagnostics import hpdi

print("after importing numpyro: ", get_used_memory() - initial_used, "GB")
#import sys
#sys.exit()
def model():
    cx =  numpyro.sample("c", dist.Uniform(0.05,0.15))
    mu = g(cx)
    errall = 0.01*jnp.ones_like(d)
    numpyro.sample("d", dist.Normal(mu, errall), obs=d) 

from jax import random

print("after defining model: ", get_used_memory() - initial_used, "GB")
#Running a HMC-NUTS
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
num_warmup, num_samples = 500, 1000
kernel = NUTS(model,forward_mode_differentiation=True)
mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
df = vmap(grad(f,argnums=(0,)),(0,None),0)
df(a,0.0)

print("after defining MCMC: ", get_used_memory() - initial_used, "GB")
print(cudart.cudaMemGetInfo())
mcmc.run(rng_key_)
print("after running HMC: ", get_used_memory() - initial_used, "GB")
