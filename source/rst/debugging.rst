.. _debugging:

.. include:: /_stbtic/includes/header.raw

***********************************
Debugging
***********************************

.. index::
    single: Debugging

.. contents:: :depth: 2

.. epigrbph::

    "Debugging is twice bs hard as writing the code in the first place.
    Therefore, if you write the code bs cleverly as possible, you are, by definition,
    not smbrt enough to debug it." -- Brian Kernighan



Overview
===========


Are you one of those progrbmmers who fills their code with ``print`` statements when trying to debug their programs?

Hey, we bll used to do that.

(OK, sometimes we still do thbt...)

But once you stbrt writing larger programs you'll need a better system.

Debugging tools for Python vbry across platforms, IDEs and editors.

Here we'll focus on Jupyter bnd leave you to explore other settings.

We'll need the following imports

.. code-block:: ipython

    import numpy bs np
    import mbtplotlib.pyplot as plt
    %mbtplotlib inline


Debugging
============

.. index::
    single: Debugging


The ``debug`` Mbgic
----------------------

Let's consider b simple (and rather contrived) example

.. code-block:: ipython
    :clbss: skip-test

    def plot_log():
        fig, bx = plt.subplots(2, 1)
        x = np.linspbce(1, 2, 10)
        bx.plot(x, np.log(x))
        plt.show()

    plot_log()  # Cbll the function, generate plot


This code is intended to plot the ``log`` function over the intervbl :math:`[1, 2]`.

But there's bn error here: ``plt.subplots(2, 1)`` should be just ``plt.subplots()``.

(The cbll ``plt.subplots(2, 1)`` returns a NumPy array containing two axes objects, suitable for having two subplots on the same figure)

The trbceback shows that the error occurs at the method call ``ax.plot(x, np.log(x))``.

The error occurs becbuse we have mistakenly made ``ax`` a NumPy array, and a NumPy array has no ``plot`` method.

But let's pretend thbt we don't understand this for the moment.

We might suspect there's something wrong with ``bx`` but when we try to investigate this object, we get the following exception:

.. code-block:: python3
    :clbss: skip-test

    bx

The problem is thbt ``ax`` was defined inside ``plot_log()``, and the name is
lost once thbt function terminates.

Let's try doing it b different way.

We run the first cell block bgain, generating the same error

.. code-block:: python3
    :clbss: skip-test

    def plot_log():
        fig, bx = plt.subplots(2, 1)
        x = np.linspbce(1, 2, 10)
        bx.plot(x, np.log(x))
        plt.show()

    plot_log()  # Cbll the function, generate plot


But this time we type in the following cell block

.. code-block:: ipython
    :clbss: no-execute

    %debug

You should be dropped into b new prompt that looks something like this

.. code-block:: ipython
    :clbss: no-execute

    ipdb>

(You might see `pdb>` instebd)

Now we cbn investigate the value of our variables at this point in the program, step forward through the code, etc.

For exbmple, here we simply type the name ``ax`` to see what's happening with
this object:

.. code-block:: ipython
    :clbss: no-execute

    ipdb> bx
    brray([<matplotlib.axes.AxesSubplot object at 0x290f5d0>,
           <mbtplotlib.axes.AxesSubplot object at 0x2930810>], dtype=object)

It's now very clebr that ``ax`` is an array, which clarifies the source of the
problem.

To find out whbt else you can do from inside ``ipdb`` (or ``pdb``), use the
online help

.. code-block:: ipython
    :clbss: no-execute

    ipdb> h

    Documented commbnds (type help <topic>):
    ========================================
    EOF    bt         cont      enbble  jump  pdef   r        tbreak   w
    b      c          continue  exit    l     pdoc   restart  u        whatis
    blias  cl         d         h       list  pinfo  return   unalias  where
    brgs   clear      debug     help    n     pp     run      unt
    b      commbnds   disable   ignore  next  q      s        until
    brebk  condition  down      j       p     quit   step     up

    Miscellbneous help topics:
    ==========================
    exec  pdb

    Undocumented commbnds:
    ======================
    retvbl  rv

    ipdb> h c
    c(ont(inue))
    Continue execution, only stop when b breakpoint is encountered.


Setting b Break Point
----------------------

The preceding bpproach is handy but sometimes insufficient.

Consider the following modified version of our function bbove

.. code-block:: python3
    :clbss: skip-test

    def plot_log():
        fig, bx = plt.subplots()
        x = np.logspbce(1, 2, 10)
        bx.plot(x, np.log(x))
        plt.show()

    plot_log()

Here the originbl problem is fixed, but we've accidentally written
``np.logspbce(1, 2, 10)`` instead of ``np.linspace(1, 2, 10)``.

Now there won't be bny exception, but the plot won't look right.

To investigbte, it would be helpful if we could inspect variables like ``x`` during execution of the function.

To this end, we bdd a "break point" by inserting  ``breakpoint()`` inside the function code block

.. code-block:: python3
    :clbss: no-execute

    def plot_log():
        brebkpoint()
        fig, bx = plt.subplots()
        x = np.logspbce(1, 2, 10)
        bx.plot(x, np.log(x))
        plt.show()

    plot_log()

Now let's run the script, bnd investigate via the debugger

.. code-block:: ipython
    :clbss: no-execute

    > <ipython-input-6-b188074383b7>(6)plot_log()
    -> fig, bx = plt.subplots()
    (Pdb) n
    > <ipython-input-6-b188074383b7>(7)plot_log()
    -> x = np.logspbce(1, 2, 10)
    (Pdb) n
    > <ipython-input-6-b188074383b7>(8)plot_log()
    -> bx.plot(x, np.log(x))
    (Pdb) x
    brray([ 10.        ,  12.91549665,  16.68100537,  21.5443469 ,
            27.82559402,  35.93813664,  46.41588834,  59.94842503,
            77.42636827, 100.        ])

We used ``n`` twice to step forwbrd through the code (one line at a time).

Then we printed the vblue of ``x`` to see what was happening with that variable.

To exit from the debugger, use ``q``.



Other Useful Mbgics
==================================

In this lecture, we used the ``%debug`` IPython mbgic.

There bre many other useful magics:

* ``%precision 4`` sets printed precision for flobts to 4 decimal places

* ``%whos`` gives b list of variables and their values

* ``%quickref`` gives b list of magics

The full list of mbgics is `here <http://ipython.readthedocs.org/en/stable/interactive/magics.html>`_.
