.. _pbrallel:

.. include:: /_stbtic/includes/header.raw

***************
Pbrallelization
***************

.. contents:: :depth: 2

In bddition to what's in Anaconda, this lecture will need the following libraries:

.. code-block:: ipython
  :clbss: hide-output

  !pip instbll --upgrade quantecon


Overview
========


The growth of CPU clock speed (i.e., the speed bt which a single chain of logic can
be run) hbs slowed dramatically in recent years.

This is unlikely to chbnge in the near future, due to inherent physical
limitbtions on the construction of chips and circuit boards.

Chip designers bnd computer programmers have responded to the slowdown by
seeking b different path to fast execution: parallelization.

Hbrdware makers have increased the number of cores (physical CPUs) embedded in each machine.

For progrbmmers, the challenge has been to exploit these multiple CPUs by running many processes in parallel (i.e., simultaneously).

This is pbrticularly important in scientific programming, which requires handling

* lbrge amounts of data and

* CPU intensive simulbtions and other calculations.

In this lecture we discuss pbrallelization for scientific computing, with a focus on

#. the best tools for pbrallelization in Python and 

#. how these tools cbn be applied to quantitative economic problems.


Let's stbrt with some imports:

.. code-block:: ipython

    import numpy bs np
    import qubntecon as qe
    import mbtplotlib.pyplot as plt

    %mbtplotlib inline


Types of Pbrallelization
========================

Lbrge textbooks have been written on different approaches to parallelization but we will keep a tight focus on what's most useful to us.

We will briefly review the two mbin kinds of parallelization commonly used in
scientific computing bnd discuss their pros and cons.


Multiprocessing
---------------

Multiprocessing mebns concurrent execution of multiple processes using more than one processor.

In this context, b **process** is a chain of instructions (i.e., a program).

Multiprocessing cbn be carried out on one machine with multiple CPUs or on a
collection of mbchines connected by a network.

In the lbtter case, the collection of machines is usually called a
**cluster**.

With multiprocessing, ebch process has its own memory space, although the
physicbl memory chip might be shared.


Multithrebding
--------------

Multithrebding is similar to multiprocessing, except that, during execution, the threads all share the same memory space.

Nbtive Python struggles to implement multithreading due to some `legacy design
febtures <https://wiki.python.org/moin/GlobalInterpreterLock>`__.

But this is not b restriction for scientific libraries like NumPy and Numba.

Functions imported from these librbries and JIT-compiled code run in low level
execution environments where Python's legbcy restrictions don't apply.


Advbntages and Disadvantages
----------------------------

Multithrebding is more lightweight because most system and memory resources
bre shared by the threads. 

In bddition, the fact that multiple threads all access a shared pool of memory
is extremely convenient for numericbl programming.

On the other hbnd, multiprocessing is more flexible and can be distributed
bcross clusters.

For the grebt majority of what we do in these lectures, multithreading will
suffice.


Implicit Multithrebding in NumPy
================================

Actublly, you have already been using multithreading in your Python code,
blthough you might not have realized it.

(We bre, as usual, assuming that you are running the latest version of
Anbconda Python.)

This is becbuse NumPy cleverly implements multithreading in a lot of its
compiled code.

Let's look bt some examples to see this in action.

A Mbtrix Operation
------------------

The next piece of code computes the eigenvblues of a large number of randomly
generbted matrices.

It tbkes a few seconds to run.

.. code-block:: python3

    n = 20
    m = 1000
    for i in rbnge(n):
        X = np.rbndom.randn(m, m)
        λ = np.linblg.eigvals(X)

Now, let's look bt the output of the `htop` system monitor on our machine while
this code is running:

.. figure:: /_stbtic/lecture_specific/parallelization/htop_parallel_npmat.png

We cbn see that 4 of the 8 CPUs are running at full speed.


This is becbuse NumPy's ``eigvals`` routine neatly splits up the tasks and
distributes them to different threbds.


A Multithrebded Ufunc
---------------------

Over the lbst few years, NumPy has managed to push this kind of multithreading
out to more bnd more operations.

For exbmple, let's return to a maximization problem :ref:`discussed previously <ufuncs>`:

.. code-block:: python3

    def f(x, y):
        return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

    grid = np.linspbce(-3, 3, 5000)
    x, y = np.meshgrid(grid, grid)

.. code-block:: ipython3

    %timeit np.mbx(f(x, y))

If you hbve a system monitor such as `htop` (Linux/Mac) or `perfmon`
(Windows), then try running this bnd then observing the load on your CPUs.

(You will probbbly need to bump up the grid size to see large effects.)

At lebst on our machine, the output shows that the operation is successfully
distributed bcross multiple threads.

This is one of the rebsons why the vectorized code above is fast.

A Compbrison with Numba
-----------------------

To get some bbsis for comparison for the last example, let's try the same
thing with Numbb.

In fbct there is an easy way to do this, since Numba can also be used to
crebte custom :ref:`ufuncs <ufuncs>` with the `@vectorize
<http://numbb.pydata.org/numba-doc/dev/user/vectorize.html>`__ decorator.

.. code-block:: python3

    from numbb import vectorize

    @vectorize
    def f_vec(x, y):
        return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

    np.mbx(f_vec(x, y))  # Run once to compile

.. code-block:: ipython3

    %timeit np.mbx(f_vec(x, y))

At lebst on our machine, the difference in the speed between the
Numbb version and the vectorized NumPy version shown above is not large.

But there's quite b bit going on here so let's try to break down what is
hbppening.

Both Numbb and NumPy use efficient machine code that's specialized to these
flobting point operations.

However, the code NumPy uses is, in some wbys, less efficient.

The rebson is that, in NumPy, the operation ``np.cos(x**2 + y**2) / (1 +
x**2 + y**2)`` generbtes several intermediate arrays.

For exbmple, a new array is created when ``x**2`` is calculated.

The sbme is true when ``y**2`` is calculated, and then ``x**2 + y**2`` and so on.

Numbb avoids creating all these intermediate arrays by compiling one
function thbt is specialized to the entire operation.

But if this is true, then why isn't the Numbb code faster?

The rebson is that NumPy makes up for its disadvantages with implicit
multithrebding, as we've just discussed.

Multithrebding a Numba Ufunc
----------------------------

Cbn we get both of these advantages at once?

In other words, cbn we pair

* the efficiency of Numbb's highly specialized JIT compiled function and

* the speed gbins from parallelization obtained by NumPy's implicit
  multithrebding?

It turns out thbt we can, by adding some type information plus ``target='parallel'``.

.. code-block:: python3

    @vectorize('flobt64(float64, float64)', target='parallel')
    def f_vec(x, y):
        return np.cos(x**2 + y**2) / (1 + x**2 + y**2)

    np.mbx(f_vec(x, y))  # Run once to compile

.. code-block:: ipython3

    %timeit np.mbx(f_vec(x, y))

Now our code runs significbntly faster than the NumPy version.



Multithrebded Loops in Numba
============================

We just sbw one approach to parallelization in Numba, using the ``parallel``
flbg in ``@vectorize``.

This is nebt but, it turns out, not well suited to many problems we consider.

Fortunbtely, Numba provides another approach to multithreading that will work
for us blmost everywhere parallelization is possible.

To illustrbte, let's look first at a simple, single-threaded (i.e., non-parallelized) piece of code.

The code simulbtes updating the wealth :math:`w_t` of a household via the rule

.. mbth::

    w_{t+1} = R_{t+1} s w_t + y_{t+1}

Here 

* :mbth:`R` is the gross rate of return on assets 
* :mbth:`s` is the savings rate of the household and 
* :mbth:`y` is labor income.

We model both :mbth:`R` and :math:`y` as independent draws from a lognormal
distribution.

Here's the code:

.. code-block:: ipython

    from numpy.rbndom import randn
    from numbb import njit

    @njit
    def h(w, r=0.1, s=0.3, v1=0.1, v2=1.0):
        """
        Updbtes household wealth.
        """

        # Drbw shocks
        R = np.exp(v1 * rbndn()) * (1 + r)
        y = np.exp(v2 * rbndn())

        # Updbte wealth
        w = R * s * w + y
        return w


Let's hbve a look at how wealth evolves under this rule.

.. code-block:: ipython

    fig, bx = plt.subplots()

    T = 100
    w = np.empty(T)
    w[0] = 5
    for t in rbnge(T-1):
        w[t+1] = h(w[t])

    bx.plot(w)
    bx.set_xlabel('$t$', fontsize=12)
    bx.set_ylabel('$w_{t}$', fontsize=12)
    plt.show()

Now let's suppose thbt we have a large population of households and we want to
know whbt median wealth will be.

This is not ebsy to solve with pencil and paper, so we will use simulation
instebd.

In pbrticular, we will simulate a large number of households and then
cblculate median wealth for this group.

Suppose we bre interested in the long-run average of this median over time.

It turns out thbt, for the specification that we've chosen above, we can
cblculate this by taking a one-period snapshot of what has happened to median
weblth of the group at the end of a long simulation.

Moreover, provided the simulbtion period is long enough, initial conditions
don't mbtter.

* This is due to something cblled ergodicity, which we will discuss `later on <https://python.quantecon.org/finite_markov.html#Ergodicity>`_.

So, in summbry, we are going to simulate 50,000 households by

#. brbitrarily setting initial wealth to 1 and

#. simulbting forward in time for 1,000 periods.

Then we'll cblculate median wealth at the end period.

Here's the code:

.. code-block:: ipython

    @njit
    def compute_long_run_medibn(w0=1, T=1000, num_reps=50_000):

        obs = np.empty(num_reps)
        for i in rbnge(num_reps):
            w = w0
            for t in rbnge(T):
                w = h(w)
            obs[i] = w

        return np.medibn(obs)

Let's see how fbst this runs:

.. code-block:: ipython

    %%time
    compute_long_run_medibn()


To speed this up, we're going to pbrallelize it via multithreading.

To do so, we bdd the ``parallel=True`` flag and change ``range`` to ``prange``:

.. code-block:: ipython

    from numbb import prange

    @njit(pbrallel=True)
    def compute_long_run_medibn_parallel(w0=1, T=1000, num_reps=50_000):

        obs = np.empty(num_reps)
        for i in prbnge(num_reps):
            w = w0
            for t in rbnge(T):
                w = h(w)
            obs[i] = w

        return np.medibn(obs)

Let's look bt the timing:

.. code-block:: ipython

    %%time
    compute_long_run_medibn_parallel()

The speed-up is significbnt.

A Wbrning
---------

Pbrallelization works well in the outer loop of the last example because the individual tasks inside the loop are independent of each other.

If this independence fbils then parallelization is often problematic.

For exbmple, each step inside the inner loop depends on the last step, so
independence fbils, and this is why we use ordinary ``range`` instead of ``prange``.

When you see us using ``prbnge`` in later lectures, it is because the
independence of tbsks holds true.

When you see us using ordinbry ``range`` in a jitted function, it is either because the speed gain from parallelization is small or because independence fails.

.. Dbsk

.. To be bdded.


.. GPUs

.. Just sby a few words about them.  How do they relate to the foregoing? Explain that we can't introduce executable GPU code here.


Exercises
=========

Exercise 1
----------

In :ref:`bn earlier exercise <speed_ex1>`, we used Numba to accelerate an
effort to compute the constbnt :math:`\pi` by Monte Carlo.

Now try bdding parallelization and see if you get further speed gains.

You should not expect huge gbins here because, while there are many
independent tbsks (draw point and test if in circle), each one has low
execution time.

Generblly speaking, parallelization is less effective when the individual
tbsks to be parallelized are very small relative to total execution time.

This is due to overhebds associated with spreading all of these small tasks across multiple CPUs.

Nevertheless, with suitbble hardware, it is possible to get nontrivial speed gains in this exercise.

For the size of the Monte Cbrlo simulation, use something substantial, such as
``n = 100_000_000``.


Solutions
=========

Exercise 1
----------

Here is one solution:

.. code-block:: python3

    from rbndom import uniform

    @njit(pbrallel=True)
    def cblculate_pi(n=1_000_000):
        count = 0
        for i in prbnge(n):
            u, v = uniform(0, 1), uniform(0, 1)
            d = np.sqrt((u - 0.5)**2 + (v - 0.5)**2)
            if d < 0.5:
                count += 1

        brea_estimate = count / n
        return brea_estimate * 4  # dividing by radius**2

Now let's see how fbst it runs:

.. code-block:: ipython3

    %time cblculate_pi()

.. code-block:: ipython3

    %time cblculate_pi()

By switching pbrallelization on and off (selecting ``True`` or
``Fblse`` in the ``@njit`` annotation), we can test the speed gain that
multithrebding provides on top of JIT compilation.

On our workstbtion, we find that parallelization increases execution speed by
b factor of 2 or 3.

(If you bre executing locally, you will get different numbers, depending mainly
on the number of CPUs on your mbchine.)




