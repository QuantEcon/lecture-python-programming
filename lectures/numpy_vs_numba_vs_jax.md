---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(numpy_numba_jax)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# NumPy vs Numba vs JAX

In the preceding lectures, we've discussed three core libraries for scientific
and numerical computing:

* [NumPy](numpy)
* [Numba](numba)
* [JAX](jax_intro)

Which one should we use in any given situation?

This lecture addresses that question, at least partially, by discussing some use cases.

Before getting started, we note that the first two are a natural pair: NumPy and
Numba play well together.

JAX, on the other hand, stands alone.

When considering each approach, we will consider not just efficiency and memory
footprint but also clarity and ease of use.

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
---
tags: [hide-output]
---
!pip install quantecon jax
```

```{include} _admonition/gpu.md
```

We will use the following imports.

```{code-cell} ipython3
from functools import partial

import numpy as np
import numba
import quantecon as qe
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import jax
import jax.numpy as jnp
from jax import lax
```

## Vectorized operations

Some operations can be perfectly vectorized --- all loops are easily eliminated
and numerical operations are reduced to calculations on arrays.

In this case, which approach is best?

### Problem Statement

Consider the problem of maximizing a function $f$ of two variables $(x,y)$ over
the square $[-a, a] \times [-a, a]$.

For $f$ and $a$ let's choose

$$
f(x,y) = \frac{\cos(x^2 + y^2)}{1 + x^2 + y^2}
\quad \text{and} \quad
a = 3
$$

Here's a plot of $f$

```{code-cell} ipython3

def f(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

xgrid = np.linspace(-3, 3, 50)
ygrid = xgrid
x, y = np.meshgrid(xgrid, ygrid)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,
                y,
                f(x, y),
                rstride=2, cstride=2,
                cmap=cm.viridis,
                alpha=0.7,
                linewidth=0.25)
ax.set_zlim(-0.5, 1.0)
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$y$', fontsize=14)
plt.show()
```

For the sake of this exercise, we're going to use brute force for the
maximization.

1. Evaluate $f$ for all $(x,y)$ in a grid on the square.
1. Return the maximum of observed values.

Just to illustrate the idea, here's a non-vectorized version that uses Python loops.

```{code-cell} ipython3
grid = np.linspace(-3, 3, 50)
m = -np.inf
for x in grid:
    for y in grid:
        z = f(x, y)
        m = max(m, z)
```


### NumPy vectorization

Let's switch to NumPy and use a larger grid

Here we use `np.meshgrid` to create two-dimensional input grids `x` and `y` such
that `f(x, y)` generates all evaluations on the product grid.


```{code-cell} ipython3
# Large grid
grid = np.linspace(-3, 3, 3_000)

x, y = np.meshgrid(grid, grid)    # MATLAB style meshgrid

with qe.Timer():
    z_max_numpy = np.max(f(x, y))
```

In the vectorized version, all the looping takes place in compiled code.

The use of `meshgrid` allows us to replicate the nested for loop.

The output should be close to one:

```{code-cell} ipython3
print(f"NumPy result: {z_max_numpy:.6f}")
```



### A Comparison with Numba

Now let's see if we can achieve better performance using Numba with a simple loop.

```{code-cell} ipython3
@numba.jit
def compute_max_numba(grid):
    m = -np.inf
    for x in grid:
        for y in grid:
            z = np.cos(x**2 + y**2) / (1 + x**2 + y**2)
            m = max(m, z)
    return m
```

Let's test it:

```{code-cell} ipython3
grid = np.linspace(-3, 3, 3_000)

with qe.Timer():
    # First run
    z_max_numba = compute_max_numba(grid)
```

Let's run again to eliminate compile time.

```{code-cell} ipython3
with qe.Timer():
    # Second run
    compute_max_numba(grid)
```

Depending on your machine, the Numba version might be either slower or faster than NumPy.

In most cases we find that Numba is slightly better.

On the one hand, NumPy combines efficient arithmetic with some
multithreading, which provides an advantage.

On the other hand, the Numba routine uses much less memory, since we are only
working with a single one-dimensional grid.


### Parallelized Numba

Now let's try parallelization with Numba using `prange`:

```{code-cell} ipython3
@numba.jit(parallel=True)
def compute_max_numba_parallel(grid):
    n = len(grid)
    m = -np.inf
    for i in numba.prange(n):
        for j in range(n):
            x = grid[i]
            y = grid[j]
            z = np.cos(x**2 + y**2) / (1 + x**2 + y**2)
            m = max(m, z)
    return m
```

Here's a warm up run and test.

```{code-cell} ipython3
with qe.Timer():
    # First run
    z_max_parallel = compute_max_numba_parallel(grid)
```

Here's the timing for the pre-compiled version.

```{code-cell} ipython3
with qe.Timer():
    # Second run
    compute_max_numba_parallel(grid)
```

If you have multiple cores, you should see benefits from parallelization here.

Let's make sure we're still getting the right result (close to one):

```{code-cell} ipython3
print(f"Numba result: {z_max_parallel:.6f}")
```


For powerful machines and larger grid sizes, parallelization can generate
useful speed gains, even on the CPU.


### Vectorized code with JAX

Let's try replicating the NumPy vectorized approach with JAX.

Let's start with the function, which switches `np` to `jnp` and adds `jax.jit`

```{code-cell} ipython3
@jax.jit
def f(x, y):
    return jnp.cos(x**2 + y**2) / (1 + x**2 + y**2)

```

We use the NumPy style meshgrid approach:

```{code-cell} ipython3
grid = jnp.linspace(-3, 3, 3_000)
x_mesh, y_mesh = jnp.meshgrid(grid, grid)
```

Now let's run and time

```{code-cell} ipython3
with qe.Timer():
    # First run
    z_max = jnp.max(f(x_mesh, y_mesh))
    # Hold interpreter
    z_max.block_until_ready()

print(f"Plain vanilla JAX result: {z_max:.6f}")
```

Let's run again to eliminate compile time.

```{code-cell} ipython3
with qe.Timer():
    # Second run
    z_max = jnp.max(f(x_mesh, y_mesh))
    # Hold interpreter
    z_max.block_until_ready()
```

Once compiled, JAX is significantly faster than NumPy, especially on a GPU.

The compilation overhead is a one-time cost that pays off when the function is called repeatedly.


### JAX plus vmap

There is one problem with both the NumPy code and the JAX code above:

While the flat arrays are low-memory

```{code-cell} ipython3
grid.nbytes 
```

the mesh grids are memory intensive

```{code-cell} ipython3
x_mesh.nbytes + y_mesh.nbytes
```

This extra memory usage can be a big problem in actual research calculations.

Fortunately, JAX admits a different approach
using [jax.vmap](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html).

The idea of `vmap` is to break vectorization into stages, transforming a
function that operates on single values into one that operates on arrays.

Here's how we can apply it to our problem.


```{code-cell} ipython3
@jax.jit
def compute_max_vmap(grid):
    # Construct a function that takes the max over all x for given y
    f_vec_x_max = lambda y: jnp.max(f(grid, y))
    # Vectorize the function so we can call on all y simultaneously
    f_vec_max = jax.vmap(f_vec_x_max)
    # Compute the max across x at every y
    maxes = f_vec_max(grid)
    # Compute the max of the maxes and return
    return jnp.max(maxes)
```

Note that we never create 

* the two-dimensional grid `x_mesh`
* the two-dimensional grid `y_mesh` or
* the two-dimensional array `f(x,y)`

And because everything is under a single `@jax.jit`, the compiler can fuse
all operations into one optimized kernel.

Let's try it.

```{code-cell} ipython3
with qe.Timer():
    # First run
    z_max = compute_max_vmap(grid)
    # Hold interpreter
    z_max.block_until_ready()

print(f"JAX vmap result: {z_max:.6f}")
```

Let's run it again to eliminate compilation time:

```{code-cell} ipython3
with qe.Timer():
    # Second run
    z_max = compute_max_vmap(grid)
    # Hold interpreter
    z_max.block_until_ready()
```


### Summary

In our view, JAX is the winner for vectorized operations.

It dominates NumPy both in terms of speed (via JIT-compilation and
parallelization) and memory efficiency (via vmap).

Moreover, the `vmap` approach can sometimes lead to significantly clearer code.

While Numba is impressive, the beauty of JAX is that, with fully vectorized
operations, we can run exactly the same code on machines with hardware
accelerators and reap all the benefits without extra effort.

Moreover, JAX already knows how to effectively parallelize many common array
operations, which is key to fast execution.

For most cases encountered in economics, econometrics, and finance, it is
far better to hand over to the JAX compiler for efficient parallelization than to
try to hand code these routines ourselves.


## Sequential operations

Some operations are inherently sequential -- and hence difficult or impossible
to vectorize.

In this case NumPy is a poor option and we are left with the choice of Numba or
JAX.

To compare these choices, we will revisit the problem of iterating on the
quadratic map that we saw in our {doc}`Numba lecture <numba>`.


### Numba Version

Here's the Numba version.

```{code-cell} ipython3
@numba.jit
def qm(x0, n, α=4.0):
    x = np.empty(n+1)
    x[0] = x0
    for t in range(n):
      x[t+1] = α * x[t] * (1 - x[t])
    return x
```

Let's generate a time series of length 10,000,000 and time the execution:

```{code-cell} ipython3
n = 10_000_000

with qe.Timer():
    # First run
    x = qm(0.1, n)
```

Let's run it again to eliminate compilation time:

```{code-cell} ipython3
with qe.Timer():
    # Second run
    x = qm(0.1, n)
```

Numba handles this sequential operation very efficiently.


### JAX Version

We cannot directly replace `numba.jit` with `jax.jit` because JAX arrays are immutable.

But we can still implement this operation

#### First Attempt

Here's a workaround using the `at[t].set` syntax we {ref}`discussed in the JAX lecture <jax_at_workaround>`.

We'll apply a `lax.fori_loop`, which is a version of a for loop that can be compiled by XLA.

```{code-cell} ipython3
cpu = jax.devices("cpu")[0]

@partial(jax.jit, static_argnames=("n",), device=cpu)
def qm_jax_fori(x0, n, α=4.0):

    x = jnp.empty(n + 1).at[0].set(x0)

    def update(t, x):
        return x.at[t + 1].set(α * x[t] * (1 - x[t]))

    x = lax.fori_loop(0, n, update, x)
    return x

```

* We hold `n` static because it affects array size and hence JAX wants to specialize on its value in the compiled code.
* We pin to the CPU via `device=cpu` because this sequential workload consists of many small operations, leaving little opportunity for GPU parallelism.

Important: Although `at[t].set` appears to create a new array at each step, inside a JIT-compiled function the compiler detects that the old array is no longer needed and performs the update in place!

Let's time it with the same parameters:

```{code-cell} ipython3
with qe.Timer():
    # First run
    x_jax = qm_jax_fori(0.1, n)
    # Hold interpreter
    x_jax.block_until_ready()
```

Let's run it again to eliminate compilation overhead:

```{code-cell} ipython3
with qe.Timer():
    # Second run
    x_jax = qm_jax_fori(0.1, n)
    # Hold interpreter
    x_jax.block_until_ready()
```

JAX is also quite efficient for this sequential operation!


#### Second Attempt

There's another way we can implement the loop that uses `lax.scan`.

This alternative is arguably more in line with JAX's functional approach ---
although the syntax is difficult to remember.


```{code-cell} ipython3
@partial(jax.jit, static_argnames=("n",), device=cpu)
def qm_jax_scan(x0, n, α=4.0):
    def update(x, t):
        x_new = α * x * (1 - x)
        return x_new, x_new

    _, x = lax.scan(update, x0, jnp.arange(n))
    return jnp.concatenate([jnp.array([x0]), x])
```

This code is not easy to read but, in essence, `lax.scan` repeatedly calls `update` and accumulates the returns `x_new` into an array.

Let's time it with the same parameters:

```{code-cell} ipython3
with qe.Timer():
    # First run
    x_jax = qm_jax_scan(0.1, n)
    # Hold interpreter
    x_jax.block_until_ready()
```

Let's run it again to eliminate compilation overhead:

```{code-cell} ipython3
with qe.Timer():
    # Second run
    x_jax = qm_jax_scan(0.1, n)
    # Hold interpreter
    x_jax.block_until_ready()
```

Surprisingly, JAX also delivers strong performance after compilation.


### Summary

While both Numba and JAX deliver strong performance for sequential operations, there are differences in code readability and ease of use.

The Numba version is straightforward and natural to read: we simply allocate an
array and fill it element by element using a standard Python loop.

This is exactly how most programmers think about the algorithm.

The JAX versions, on the other hand, require either `lax.fori_loop` or
`lax.scan`, both of which are less intuitive than a standard Python loop.

While JAX's `at[t].set` syntax does allow element-wise updates, the overall code
remains harder to read than the Numba equivalent.

For this type of sequential operation, Numba is the clear winner in terms of
code clarity and ease of implementation.


## Overall recommendations

Let's now step back and summarize the trade-offs.

For **vectorized operations**, JAX is the strongest choice.

It matches or exceeds NumPy in speed, thanks to JIT compilation and efficient
parallelization across CPUs and GPUs.

The `vmap` transformation reduces memory usage and often leads to clearer code
than traditional meshgrid-based vectorization.

In addition, JAX functions are automatically differentiable, as we explore in
{doc}`autodiff`.

For **sequential operations**, Numba has clear advantages.

The code is natural and readable --- just a Python loop with a decorator ---
and performance is excellent.

JAX can handle sequential problems via `lax.fori_loop` or `lax.scan`, but
the syntax is less intuitive.

```{note}
One important advantage of `lax.fori_loop` and `lax.scan` is that they
support automatic differentiation through the loop, which Numba cannot do.
If you need to differentiate through a sequential computation (e.g., computing
sensitivities of a trajectory to model parameters), JAX is the better choice
despite the less natural syntax.
```

In practice, many problems involve a mix of both patterns.

A good rule of thumb: default to JAX for new projects, especially when
hardware acceleration or differentiability might be useful, and reach for Numba
when you have a tight sequential loop that needs to be fast and readable.

