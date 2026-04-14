---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# JAX

This lecture provides a short introduction to [Google JAX](https://github.com/jax-ml/jax).

```{include} _admonition/gpu.md
```

JAX is a high-performance scientific computing library that provides

* a [NumPy](https://en.wikipedia.org/wiki/NumPy)-like interface that can automatically parallelize across CPUs and GPUs,
* a just-in-time compiler for accelerating a large range of numerical
  operations, and
* [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation).

Increasingly, JAX also maintains and provides [more specialized scientific
computing routines](https://docs.jax.dev/en/latest/jax.scipy.html), such as those originally found in [SciPy](https://en.wikipedia.org/wiki/SciPy).

In addition to what's in Anaconda, this lecture will need the following libraries:

```{code-cell} ipython3
:tags: [hide-output]

!pip install jax quantecon
```

We'll use the following imports

```{code-cell} ipython3
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import quantecon as qe
```



## JAX as a NumPy Replacement

Let's look at the similarities and differences between JAX and NumPy.

### Similarities

Above we import `jax.numpy as jnp`, which provides a NumPy-like interface to
array operations.

One of the attractive features of JAX is that, whenever possible, this interface
conform to the NumPy API.

As a result, we can often use JAX as a drop-in NumPy replacement.

Here are some standard array operations using `jnp`:

```{code-cell} ipython3
a = jnp.asarray((1.0, 3.2, -1.5))
```

```{code-cell} ipython3
print(a)
```

```{code-cell} ipython3
print(jnp.sum(a))
```

```{code-cell} ipython3
print(jnp.dot(a, a))
```

It should be remembered, however, that the array object `a` is not a NumPy array:

```{code-cell} ipython3
a
```

```{code-cell} ipython3
type(a)
```

Even scalar-valued maps on arrays return JAX arrays rather than scalars!

```{code-cell} ipython3
jnp.sum(a)
```



### Differences

Let's now look at some differences between JAX and NumPy array operations.

(jax_speed)=
#### Speed!

One major difference is that JAX is faster --- and sometimes much faster.

To illustrate, suppose that we want to evaluate the cosine function at many points.

```{code-cell}
n = 50_000_000
x = np.linspace(0, 10, n)   # NumPy array
```

##### With NumPy

Let's try with NumPy

```{code-cell}
with qe.Timer():
    # First NumPy timing
    y = np.cos(x)
```

And one more time.

```{code-cell}
with qe.Timer():
    # Second NumPy timing
    y = np.cos(x)
```

Here 

* NumPy uses a pre-built binary for applying cosine to an array of floats
* The binary runs on the local machine's CPU


##### With JAX

Now let's try with JAX.

```{code-cell}
x = jnp.linspace(0, 10, n)
```

Let's time the same procedure.

```{code-cell}
with qe.Timer():
    # First run
    y = jnp.cos(x)
    # Hold the interpreter until the array operation finishes
    y.block_until_ready()
```

```{note}
Above, the `block_until_ready` method
holds the interpreter until the results of the computation are returned.
This is necessary for timing execution because JAX uses asynchronous dispatch, which
allows the Python interpreter to run ahead of numerical computations.
```

Now let's time it again.

```{code-cell}
with qe.Timer():
    # Second run
    y = jnp.cos(x)
    # Hold interpreter 
    y.block_until_ready()
```

On a GPU, this code runs much faster than its NumPy equivalent.

Also, typically, the second run is faster than the first due to JIT compilation.

This is because even built in functions like `jnp.cos` are JIT-compiled --- and the
first run includes compile time.

Why would JAX want to JIT-compile built in functions like `jnp.cos` instead of
just providing pre-compiled versions, like NumPy?

The reason is that the JIT compiler wants to specialize on the *size* of the array
being used (as well as the data type).

The size matters for generating optimized code because efficient parallelization
requires matching the size of the task to the available hardware.


#### Size Experiment

We can verify the claim that JAX specializes on array size by changing the input
size and watching the runtimes. 

```{code-cell}
x = jnp.linspace(0, 10, n + 1)
```

```{code-cell}
with qe.Timer():
    # First run
    y = jnp.cos(x)
    # Hold interpreter
    y.block_until_ready()
```


```{code-cell}
with qe.Timer():
    # Second run
    y = jnp.cos(x)
    # Hold interpreter
    y.block_until_ready()
```

The run time increases and then falls again (this will be more obvious on the GPU).

This is in line with the discussion above -- the first run after changing array
size shows compilation overhead.

Further discussion of JIT compilation is provided below.



#### Precision

Another difference between NumPy and JAX is that JAX uses 32 bit floats by default.

This is because JAX is often used for GPU computing, and most GPU computations use 32 bit floats.

Using 32 bit floats can lead to significant speed gains with small loss of precision.

However, for some calculations precision matters.

In these cases 64 bit floats can be enforced via the command 

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```

Let's check this works:

```{code-cell} ipython3
jnp.ones(3)
```


#### Immutability

As a NumPy replacement, a more significant difference is that arrays are treated as **immutable**.

For example, with NumPy we can write

```{code-cell} ipython3
a = np.linspace(0, 1, 3)
a
```

and then mutate the data in memory:

```{code-cell} ipython3
a[0] = 1
a
```

In JAX this fails 😱.


```{code-cell} ipython3
a = jnp.linspace(0, 1, 3)
a
```

```{code-cell} ipython3
try:
    a[0] = 1
except Exception as e:
    print(e)

```

The designers of JAX chose to make arrays immutable because 

1. JAX uses a *functional programming style* and
2. functional programming typically avoids mutable data

We discuss these ideas {ref}`below <jax_func>`.


(jax_at_workaround)=
#### A Workaround

JAX does provide a direct alternative to in-place array modification
via the [`at` method](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html).

```{code-cell} ipython3
a = jnp.linspace(0, 1, 3)
```

Applying `at[0].set(1)` returns a new copy of `a` with the first element set to 1

```{code-cell} ipython3
a = a.at[0].set(1)
a
```

Obviously, there are downsides to using `at`:

* The syntax is cumbersome and 
* we want to avoid creating fresh arrays in memory every time we change a single value!

Hence, for the most part, we try to avoid this syntax.

(Although it can in fact be efficient inside JIT-compiled functions -- but let's put this aside for now.)


(jax_func)=
## Functional Programming

From JAX's documentation:

*When walking about the countryside of Italy, the people will not hesitate to tell you that JAX has "una anima di pura programmazione funzionale".*

In other words, JAX assumes a 
[functional programming](https://en.wikipedia.org/wiki/Functional_programming)
style.

### Pure functions

The major implication is that JAX functions should be pure.

[Pure functions](https://en.wikipedia.org/wiki/Pure_function) have the following characteristics:

1. *Deterministic*
2. *No side effects*

[Deterministic](https://en.wikipedia.org/wiki/Deterministic_algorithm) means

*  Same input $\implies$ same output
*  Outputs do not depend on global state

In particular, pure functions will always return the same result if invoked with the same inputs.

[No side effects](https://en.wikipedia.org/wiki/Side_effect_(computer_science)) means that the function

* Won't change global state
* Won't modify data passed to the function (immutable data)



### Examples -- Pure and Impure

Here's an example of a *impure* function

```{code-cell} ipython3
tax_rate = 0.1

def add_tax(prices):
    for i, price in enumerate(prices):
        prices[i] = price * (1 + tax_rate)

prices = [10.0, 20.0]
add_tax(prices)
prices
```

This function fails to be pure because

* side effects --- it modifies the global variable `prices`
* non-deterministic --- a change to the global variable `tax_rate` will modify
  function outputs, even with the same input array `prices`.

Here's a *pure* version

```{code-cell} ipython3

def add_tax_pure(prices, tax_rate):
    new_prices = [price * (1 + tax_rate) for price in prices]
    return new_prices

tax_rate = 0.1
prices = (10.0, 20.0)
after_tax_prices = add_tax_pure(prices, tax_rate)
after_tax_prices
```

This is pure because 

* all dependencies explicit through function arguments
* and doesn't modify any external state 


### Why Functional Programming?

At QuantEcon we love pure functions because they

* Help testing: each function can operate in isolation
* Promote deterministic behavior and hence reproducibility
* Prevent bugs that arise from mutating shared state

The JAX compiler loves pure functions and functional programming because

* Data dependencies are explicit, which helps with optimizing complex computations
* Pure functions are easier to differentiate (autodiff)
* Pure functions are easier to parallelize and optimize (don't depend on shared mutable state)

Another way to think of this is as follows:

JAX represents functions as computational graphs, which are then compiled or
transformed (e.g., differentiated)

These computational graphs describe how a given set of inputs is transformed into an output.

JAX's computational graphs are pure by construction.

JAX uses a functional programming style so that user-built functions map
directly into the graph-theoretic representations supported by JAX.


## Random numbers

Random number generation in JAX differs significantly from the patterns found in NumPy or MATLAB.



### NumPy / MATLAB Approach

In NumPy / MATLAB, generation works by maintaining hidden global state.

```{code-cell} ipython3
np.random.seed(42)
print(np.random.randn(2))   
```

Each time we call a random function, the hidden state is updated:

```{code-cell} ipython3
print(np.random.randn(2)) 
```

This function is *not pure* because:

* It's non-deterministic: same inputs, different outputs
* It has side effects: it modifies the global random number generator state

This is dangerous under parallelization --- must carefully control what happens in each
thread.


### JAX


In JAX, the state of the random number generator is controlled explicitly.

First we produce a key, which seeds the random number generator.

```{code-cell} ipython3
seed = 1234
key = jax.random.key(seed)
```

Now we can use the key to generate some random numbers:

```{code-cell} ipython3
x = jax.random.normal(key, (3, 3))
x
```

If we use the same key again, we initialize at the same seed, so the random numbers are the same:

```{code-cell} ipython3
jax.random.normal(key, (3, 3))
```

To produce a (quasi-) independent draw, one option is to "split" the existing key:

```{code-cell} ipython3
key, subkey = jax.random.split(key)
```

```{code-cell} ipython3
jax.random.normal(key, (3, 3))
```

```{code-cell} ipython3
jax.random.normal(subkey, (3, 3))
```

The following diagram illustrates how `split` produces a tree of keys from a
single root, with each key generating independent random draws.

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')
ax.axis('off')

box_style = dict(boxstyle="round,pad=0.3", facecolor="white",
                 edgecolor="black", linewidth=1.5)
box_used = dict(boxstyle="round,pad=0.3", facecolor="#d4edda",
                edgecolor="black", linewidth=1.5)

# Root key
ax.text(3, 3, "key₀", ha='center', va='center', fontsize=11,
        bbox=box_style)

# Level 1
ax.annotate("", xy=(1.5, 2), xytext=(3, 2.7),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(4.5, 2), xytext=(3, 2.7),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(1.5, 2, "key₁", ha='center', va='center', fontsize=11,
        bbox=box_style)
ax.text(4.5, 2, "subkey₁", ha='center', va='center', fontsize=11,
        bbox=box_used)
ax.text(5.7, 2, "→ draw", ha='left', va='center', fontsize=10,
        color='green')

# Label the split
ax.text(2, 2.65, "split", ha='center', va='center', fontsize=9,
        fontstyle='italic', color='gray')

# Level 2
ax.annotate("", xy=(0.5, 1), xytext=(1.5, 1.7),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(2.5, 1), xytext=(1.5, 1.7),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(0.5, 1, "key₂", ha='center', va='center', fontsize=11,
        bbox=box_style)
ax.text(2.5, 1, "subkey₂", ha='center', va='center', fontsize=11,
        bbox=box_used)
ax.text(3.7, 1, "→ draw", ha='left', va='center', fontsize=10,
        color='green')

ax.text(0.7, 1.65, "split", ha='center', va='center', fontsize=9,
        fontstyle='italic', color='gray')

# Level 3
ax.annotate("", xy=(0, 0), xytext=(0.5, 0.7),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(1.5, 0), xytext=(0.5, 0.7),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(0, 0, "key₃", ha='center', va='center', fontsize=11,
        bbox=box_style)
ax.text(1.5, 0, "subkey₃", ha='center', va='center', fontsize=11,
        bbox=box_used)
ax.text(2.7, 0, "→ draw", ha='left', va='center', fontsize=10,
        color='green')
ax.text(0, 0.65, "split", ha='center', va='center', fontsize=9,
        fontstyle='italic', color='gray')

ax.text(3, -0.5, "⋮", ha='center', va='center', fontsize=14)

ax.set_title("PRNG Key Splitting Tree", fontsize=13, pad=10)
plt.tight_layout()
plt.show()
```

This syntax will seem unusual for a NumPy or Matlab user --- but will make more
sense when we get to parallel programming.

The function below produces `k` (quasi-) independent random `n x n` matrices using `split`.

```{code-cell} ipython3
def gen_random_matrices(
        key,   # JAX key for random numbers
        n=2,   # Matrices will be n x n
        k=3    # Number of matrices to generate
    ):
    matrices = []
    for _ in range(k):
        key, subkey = jax.random.split(key)
        A = jax.random.uniform(subkey, (n, n))
        matrices.append(A)
    return matrices
```

```{code-cell} ipython3
seed = 42
key = jax.random.key(seed)
gen_random_matrices(key)
```

This function is *pure*

* Deterministic: same inputs, same output
* No side effects: no hidden state is modified


### Benefits

As mentioned above, this explicitness is valuable:

* Reproducibility: Easy to reproduce results by reusing keys
* Parallelization: Control what happens on separate threads 
* Debugging: No hidden state makes code easier to test
* JIT compatibility: The compiler can optimize pure functions more aggressively


## JIT Compilation

The JAX just-in-time (JIT) compiler accelerates execution by generating
efficient machine code that varies with both task size and hardware.

We saw the power of JAX's JIT compiler combined with parallel hardware when we
{ref}`above <jax_speed>`, when we applied `cos` to a large array.

Here we study JIT compilation for more complex functions


### With NumPy

We'll try first with NumPy, using

```{code-cell}
def f(x):
    y = np.cos(2 * x**2) + np.sqrt(np.abs(x)) + 2 * np.sin(x**4) - x**2
    return y
```

Let's run with large `x`

```{code-cell}
n = 50_000_000
x = np.linspace(0, 10, n)
```

```{code-cell}
with qe.Timer():
    # Time NumPy code
    y = f(x)
```

**Eager** execution model

* Each operation is executed immediately as it is encountered, materializing its
  result before the next operation begins.

Disadvantages

* Minimal parallelization 
* Heavy memory footprint --- produces many intermediate arrays
* Lots of memory read/write



### With JAX

As a first pass, we replace `np` with `jnp` throughout:

```{code-cell}
def f(x):
    y = jnp.cos(2 * x**2) + jnp.sqrt(jnp.abs(x)) + 2 * jnp.sin(x**4) - x**2
    return y


x = jnp.linspace(0, 10, n)
```

Now let's time it.

```{code-cell}
with qe.Timer():
    # First call
    y = f(x)
    # Hold interpreter
    jax.block_until_ready(y);
```

```{code-cell}
with qe.Timer():
    # Second call
    y = f(x)
    # Hold interpreter
    jax.block_until_ready(y);
```

The outcome is similar to the `cos` example --- JAX is faster, especially on the
second run after JIT compilation.

This is because the individual array operations are parallelized on the GPU

But we are still using eager execution 

* lots of memory due to intermediate arrays
* lots of memory read/writes

Also, many separate kernels launched on the GPU

### Compiling the Whole Function

Fortunately, with JAX, we have another trick up our sleeve --- we can JIT-compile
the entire function, not just individual operations.

The compiler fuses all array operations into a single optimized kernel

Let's try this with the function `f`:

```{code-cell}
f_jax = jax.jit(f)
```

```{code-cell}
with qe.Timer():
    # First run
    y = f_jax(x)
    # Hold interpreter
    jax.block_until_ready(y);
```

```{code-cell}
with qe.Timer():
    # Second run
    y = f_jax(x)
    # Hold interpreter
    jax.block_until_ready(y);
```

The runtime has improved again --- now because we fused all the operations

* Aggressive optimization based on entire computational sequence
* Eliminates multiple calls to the hardware accelerator 

The memory footprint is also much lower --- no creation of intermediate arrays

Incidentally, a more common syntax when targeting a function for the JIT
compiler is

```{code-cell} ipython3
@jax.jit
def f(x):
    pass # put function body here
```

### How JIT compilation works

When we apply `jax.jit` to a function, JAX *traces* it: instead of executing
the operations immediately, it records the sequence of operations as a
computational graph and hands that graph to the
[XLA](https://openxla.org/xla) compiler.

XLA then fuses and optimizes the operations into a single compiled kernel
tailored to the available hardware (CPU, GPU, or TPU).

The first call to a JIT-compiled function incurs compilation overhead, but
subsequent calls with the same input shapes and types reuse the cached
compiled code and run at full speed.


### Compiling non-pure functions

While JAX will not usually throw errors when compiling impure functions,
execution becomes unpredictable!

Here's an illustration of this fact:

```{code-cell} ipython3
a = 1  # global

@jax.jit
def f(x):
    return a + x
```

```{code-cell} ipython3
x = jnp.ones(2)
```

```{code-cell} ipython3
f(x)
```

In the code above, the global value `a=1` is fused into the jitted function.

Even if we change `a`, the output of `f` will not be affected --- as long as the same compiled version is called.

```{code-cell} ipython3
a = 42
```

```{code-cell} ipython3
f(x)
```

Changing the dimension of the input triggers a fresh compilation of the function, at which time the change in the value of `a` takes effect:

```{code-cell} ipython3
x = jnp.ones(3)
```

```{code-cell} ipython3
f(x)
```

Moral of the story: write pure functions when using JAX!




## Vectorization with `vmap`

Another powerful JAX transformation is `jax.vmap`, which automatically
vectorizes a function written for a single input so that it operates over
batches.

This avoids the need to manually write vectorized code or use explicit loops.

### A simple example

Suppose we have a function that computes the difference between mean and median for an array of numbers.

```{code-cell} ipython3
def mm_diff(x):
    return jnp.mean(x) - jnp.median(x)
```

We can apply it to a single vector:

```{code-cell} ipython3
x = jnp.array([1.0, 2.0, 5.0])
mm_diff(x)
```

Now suppose we have a matrix and want to compute these statistics for each row.

Without `vmap`, we'd need an explicit loop:

```{code-cell} ipython3
X = jnp.array([[1.0, 2.0, 5.0],
               [4.0, 5.0, 6.0],
               [1.0, 8.0, 9.0]])

for row in X:
    print(mm_diff(row))
```

However, Python loops are slow and cannot be efficiently compiled or
parallelized by JAX.

With `vmap`, we can avoid loops and keep the computation on the accelerator:

```{code-cell} ipython3
batch_mm_diff = jax.vmap(mm_diff)    # Create a new "vectorized" version
batch_mm_diff(X)                     # Apply to each row of X
```


### Combining transformations

One of JAX's strengths is that transformations compose naturally.

For example, we can JIT-compile a vectorized function:

```{code-cell} ipython3
fast_batch_mm_diff = jax.jit(jax.vmap(mm_diff))
fast_batch_mm_diff(X)
```

This composition of `jit`, `vmap`, and (as we'll see next) `grad` is central to
JAX's design and makes it especially powerful for scientific computing and
machine learning.


## Automatic differentiation: a preview

JAX can use automatic differentiation to compute gradients.

This can be extremely useful for optimization and solving nonlinear systems.

Here's a simple illustration involving the function $f(x) = x^2 / 2$:

```{code-cell} ipython3
def f(x):
    return (x**2) / 2

f_prime = jax.grad(f)
```

```{code-cell} ipython3
f_prime(10.0)
```

Let's plot the function and derivative, noting that $f'(x) = x$.

```{code-cell} ipython3
fig, ax = plt.subplots()
x_grid = jnp.linspace(-4, 4, 200)
ax.plot(x_grid, f(x_grid), label="$f$")
ax.plot(x_grid, [f_prime(x) for x in x_grid], label="$f'$")
ax.legend(loc='upper center')
plt.show()
```

Automatic differentiation is a deep topic with many applications in economics
and finance.  We provide a more thorough treatment in {doc}`our lecture on
autodiff <autodiff>`.


## Exercises


```{exercise-start}
:label: jax_intro_ex2
```

In the Exercise section of {doc}`our lecture on Numba <numba>`, we {ref}`used Monte
Carlo to price a European call option <numba_ex4>`.

The code was accelerated by Numba-based multithreading.

Try writing a version of this operation for JAX, using all the same
parameters.



```{exercise-end}
```


```{solution-start} jax_intro_ex2
:class: dropdown
```
Here is one solution:

```{code-cell} ipython3
M = 10_000_000

n, β, K = 20, 0.99, 100
μ, ρ, ν, S0, h0 = 0.0001, 0.1, 0.001, 10, 0

@jax.jit
def compute_call_price_jax(β=β,
                           μ=μ,
                           S0=S0,
                           h0=h0,
                           K=K,
                           n=n,
                           ρ=ρ,
                           ν=ν,
                           M=M,
                           key=jax.random.key(1)):

    s = jnp.full(M, np.log(S0))
    h = jnp.full(M, h0)

    def update(i, loop_state):
        s, h, key = loop_state
        key, subkey = jax.random.split(key)
        Z = jax.random.normal(subkey, (2, M))
        s = s + μ + jnp.exp(h) * Z[0, :]
        h = ρ * h + ν * Z[1, :]
        new_loop_state = s, h, key
        return new_loop_state

    initial_loop_state = s, h, key
    final_loop_state = jax.lax.fori_loop(0, n, update, initial_loop_state)
    s, h, key = final_loop_state

    expectation = jnp.mean(jnp.maximum(jnp.exp(s) - K, 0))

    return β**n * expectation
```

```{note}
We use `jax.lax.fori_loop` instead of a Python `for` loop.
This allows JAX to compile the loop efficiently without unrolling it,
which significantly reduces compilation time for large arrays.
```

Let's run it once to compile it:

```{code-cell} ipython3
with qe.Timer():
    compute_call_price_jax().block_until_ready()
```

And now let's time it:

```{code-cell} ipython3
with qe.Timer():
    compute_call_price_jax().block_until_ready()
```

```{solution-end}
```
