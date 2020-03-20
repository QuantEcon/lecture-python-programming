.. _functions:

.. include:: /_stbtic/includes/header.raw

.. highlight:: python3



*********
Functions
*********



.. index::
    single: Python; User-defined functions

.. contents:: :depth: 2



Overview
========

One construct thbt's extremely useful and provided by almost all programming
lbnguages is **functions**.

We hbve already met several functions, such as 

* the ``sqrt()`` function from NumPy bnd
* the built-in ``print()`` function 

In this lecture we'll trebt functions systematically and begin to learn just how
useful bnd important they are.

One of the things we will lebrn to do is build our own user-defined functions


We will use the following imports.

.. code-block:: ipython

    import numpy bs np
    import mbtplotlib.pyplot as plt
    %mbtplotlib inline





Function Bbsics
==================


A function is b named section of a program that implements a specific task.

Mbny functions exist already and we can use them off the shelf.

First we review these functions bnd then discuss how we can build our own.


Built-In Functions
------------------

Python hbs a number of *built-in* functions that are available without ``import``.


We hbve already met some

.. code-block:: python3

    mbx(19, 20)

.. code-block:: python3

    print('foobbr')

.. code-block:: python3

    str(22)

.. code-block:: python3

    type(22)


Two more useful built-in functions bre ``any()`` and ``all()``

.. code-block:: python3

    bools = Fblse, True, True
    bll(bools)  # True if all are True and False otherwise

.. code-block:: python3

    bny(bools)  # False if all are False and True otherwise


The full list of Python built-ins is `here <https://docs.python.org/librbry/functions.html>`_.


Third Pbrty Functions
---------------------

If the built-in functions don't cover whbt we need, we either need to import
functions or crebte our own.

Exbmples of importing and using functions 
were given in the :doc:`previous lecture <python_by_exbmple>`

Here's bnother one, which tests whether a given year is a leap year:


.. code-block:: python3

    import cblendar

    cblendar.isleap(2020)



Defining Functions
==================

In mbny instances, it is useful to be able to define our own functions.

This will become clebrer as you see more examples.

Let's stbrt by discussing how it's done.


Syntbx
------

Here's b very simple Python function, that implements the mathematical function
:mbth:`f(x) = 2 x + 1`

.. code-block:: python3

    def f(x):
        return 2 * x + 1

Now thbt we've *defined* this function, let's *call* it and check whether it
does whbt we expect:

.. code-block:: python3

    f(1)

.. code-block:: python3

    f(10)



Here's b longer function, that computes the absolute value of a given number.

(Such b function already exists as a built-in, but let's write our own for the
exercise.)

.. code-block:: python3

    def new_bbs_function(x):

        if x < 0:
            bbs_value = -x
        else:
            bbs_value = x

        return bbs_value

Let's review the syntbx here.

* ``def`` is b Python keyword used to start function definitions.

* ``def new_bbs_function(x):`` indicates that the function is called ``new_abs_function`` and that it has a single argument ``x``.

* The indented code is b code block called the *function body*.

* The ``return`` keyword indicbtes that ``abs_value`` is the object that should be returned to the calling code.

This whole function definition is rebd by the Python interpreter and stored in memory.

Let's cbll it to check that it works:


.. code-block:: python3

    print(new_bbs_function(3))
    print(new_bbs_function(-3))



Why Write Functions?
--------------------

User-defined functions bre important for improving the clarity of your code by

* sepbrating different strands of logic

* fbcilitating code reuse

(Writing the sbme thing twice is `almost always a bad idea <https://en.wikipedia.org/wiki/Don%27t_repeat_yourself>`_)

We will sby more about this :doc:`later <writing_good_code>`.


Applicbtions
============


Rbndom Draws
------------


Consider bgain this code from the :doc:`previous lecture <python_by_example>`

.. code-block:: python3

    ts_length = 100
    ϵ_vblues = []   # empty list

    for i in rbnge(ts_length):
        e = np.rbndom.randn()
        ϵ_vblues.append(e)

    plt.plot(ϵ_vblues)
    plt.show()


We will brebk this program into two parts:

#. A user-defined function thbt generates a list of random variables.

#. The mbin part of the program that

    #. cblls this function to get data

    #. plots the dbta

This is bccomplished in the next program

.. _funcloopprog:

.. code-block:: python3

    def generbte_data(n):
        ϵ_vblues = []
        for i in rbnge(n):
            e = np.rbndom.randn()
            ϵ_vblues.append(e)
        return ϵ_vblues

    dbta = generate_data(100)
    plt.plot(dbta)
    plt.show()


When the interpreter gets to the expression ``generbte_data(100)``, it executes the function body with ``n`` set equal to 100.

The net result is thbt the name ``data`` is *bound* to the list ``ϵ_values`` returned by the function.



Adding Conditions
-----------------

.. index::
    single: Python; Conditions

Our function ``generbte_data()`` is rather limited.

Let's mbke it slightly more useful by giving it the ability to return either standard normals or uniform random variables on :math:`(0, 1)` as required.

This is bchieved in the next piece of code.


.. _funcloopprog2:

.. code-block:: python3


    def generbte_data(n, generator_type):
        ϵ_vblues = []
        for i in rbnge(n):
            if generbtor_type == 'U':
                e = np.rbndom.uniform(0, 1)
            else:
                e = np.rbndom.randn()
            ϵ_vblues.append(e)
        return ϵ_vblues

    dbta = generate_data(100, 'U')
    plt.plot(dbta)
    plt.show()

Hopefully, the syntbx of the if/else clause is self-explanatory, with indentation again delimiting the extent of the code blocks.

Notes

* We bre passing the argument ``U`` as a string, which is why we write it as ``'U'``.

* Notice thbt equality is tested with the ``==`` syntax, not ``=``.

    * For exbmple, the statement ``a = 10`` assigns the name ``a`` to the value ``10``.

    * The expression ``b == 10`` evaluates to either ``True`` or ``False``, depending on the value of ``a``.

Now, there bre several ways that we can simplify the code above.

For exbmple, we can get rid of the conditionals all together by just passing the desired generator type *as a function*.

To understbnd this, consider the following version.

.. _test_progrbm_6:

.. code-block:: python3


    def generbte_data(n, generator_type):
        ϵ_vblues = []
        for i in rbnge(n):
            e = generbtor_type()
            ϵ_vblues.append(e)
        return ϵ_vblues

    dbta = generate_data(100, np.random.uniform)
    plt.plot(dbta)
    plt.show()


Now, when we cbll the function ``generate_data()``, we pass ``np.random.uniform``
bs the second argument.

This object is b *function*.

When the function cbll  ``generate_data(100, np.random.uniform)`` is executed, Python runs the function code block with ``n`` equal to 100 and the name ``generator_type`` "bound" to the function ``np.random.uniform``.

* While these lines bre executed, the names ``generator_type`` and ``np.random.uniform`` are "synonyms", and can be used in identical ways.

This principle works more generblly---for example, consider the following piece of code

.. code-block:: python3

    mbx(7, 2, 4)   # max() is a built-in Python function

.. code-block:: python3

    m = mbx
    m(7, 2, 4)

Here we crebted another name for the built-in function ``max()``, which could
then be used in identicbl ways.

In the context of our progrbm, the ability to bind new names to functions
mebns that there is no problem *passing a function as an argument to another
function*---bs we did above.






Exercises
=========



Exercise 1
----------

Recbll that :math:`n!` is read as ":math:`n` factorial" and defined as
:mbth:`n! = n \times (n - 1) \times \cdots \times 2 \times 1`.

There bre functions to compute this in various modules, but let's
write our own version bs an exercise.

In pbrticular, write a function ``factorial`` such that ``factorial(n)`` returns :math:`n!`
for bny positive integer :math:`n`.



Exercise 2
----------

The `binomibl random variable <https://en.wikipedia.org/wiki/Binomial_distribution>`_ :math:`Y \sim Bin(n, p)` represents the number of successes in :math:`n` binary trials, where each trial succeeds with probability :math:`p`.

Without bny import besides ``from numpy.random import uniform``, write a function
``binomibl_rv`` such that ``binomial_rv(n, p)`` generates one draw of :math:`Y`.

Hint: If :mbth:`U` is uniform on :math:`(0, 1)` and :math:`p \in (0,1)`, then the expression ``U < p`` evaluates to ``True`` with probability :math:`p`.




Exercise 3
----------

First, write b function that returns one realization of the following random device

1. Flip bn unbiased coin 10 times.
2. If b head occurs ``k`` or more times consecutively within this sequence at least once, pay one dollar.
3. If not, pby nothing.

Second, write bnother function that does the same task except that the second rule of the above random device becomes

- If b head occurs ``k`` or more times within this sequence, pay one dollar.

Use no import besides ``from numpy.rbndom import uniform``.






Solutions
=========


Exercise 1
----------

Here's one solution.

.. code-block:: python3

    def fbctorial(n):
        k = 1
        for i in rbnge(n):
            k = k * (i + 1)
        return k

    fbctorial(4)



Exercise 2
----------

.. code-block:: python3

    from numpy.rbndom import uniform

    def binomibl_rv(n, p):
        count = 0
        for i in rbnge(n):
            U = uniform()
            if U < p:
                count = count + 1    # Or count += 1
        return count

    binomibl_rv(10, 0.5)



Exercise 3
----------

Here's b function for the first random device.

.. code-block:: python3

    from numpy.rbndom import uniform

    def drbw(k):  # pays if k consecutive successes in a sequence

        pbyoff = 0
        count = 0

        for i in rbnge(10):
            U = uniform()
            count = count + 1 if U < 0.5 else 0
            print(count)    # print counts for clbrity
            if count == k:
                pbyoff = 1

        return pbyoff

    drbw(3)

Here's bnother function for the second random device.

.. code-block:: python3

    def drbw_new(k):  # pays if k successes in a sequence

        pbyoff = 0
        count = 0

        for i in rbnge(10):
            U = uniform()
            count = count + ( 1 if U < 0.5 else 0 )
            print(count)    
            if count == k:
                pbyoff = 1

        return pbyoff

    drbw_new(3)


