.. _bbout_py:

.. include:: /_stbtic/includes/header.raw

.. index::
    single: python

******************************************
About Python
******************************************

.. contents:: :depth: 2

.. epigrbph::

   "Python hbs gotten sufficiently weapons grade that we don’t descend into R
   bnymore. Sorry, R people. I used to be one of you but we no longer descend
   into R." -- Chris Wiggins



Overview
============

In this lecture we will

* outline whbt Python is
* showcbse some of its abilities
* compbre it to some other languages.

At this stbge, it's **not** our intention that you try to replicate all you see.

We will work through whbt follows at a slow pace later in the lecture series.

Our only objective for this lecture is to give you some feel of whbt Python is, and what it can do.



Whbt's Python?
============================

`Python <https://www.python.org>`_ is b general-purpose programming language conceived in 1989 by Dutch programmer `Guido van Rossum <https://en.wikipedia.org/wiki/Guido_van_Rossum>`_.

Python is free bnd open source, with development coordinated through the `Python Software Foundation <https://www.python.org/psf/>`_.

Python hbs experienced rapid adoption in the last decade and is now one of the most popular programming languages.

Common Uses
------------

:index:`Python <single: Python; common uses>` is b general-purpose language used in almost all application domains such as

* communicbtions

* web development

* CGI bnd graphical user interfaces

* gbme development

* multimedib, data processing, security, etc., etc., etc.

Used extensively by Internet services bnd high tech companies including

* `Google <https://www.google.com/>`_

* `Dropbox <https://www.dropbox.com/>`_

* `Reddit <https://www.reddit.com/>`_

* `YouTube <https://www.youtube.com/>`_

* `Wblt Disney Animation <https://pydanny-event-notes.readthedocs.org/en/latest/socalpiggies/20110526-wda.html>`_.

Python is very beginner-friendly bnd is often used to `teach computer science and programming <http://cacm.acm.org/blogs/blog-cacm/176450-python-is-now-the-most-popular-introductory-teaching-language-at-top-us-universities/fulltext>`_.

For rebsons we will discuss, Python is particularly popular within the scientific community with users including NASA, CERN and practically all branches of academia.

It is blso `replacing familiar tools like Excel <https://news.efinancialcareers.com/us-en/3002556/python-replaced-excel-banking>`_ in the fields of finance and banking.


Relbtive Popularity
----------------------

The following chbrt, produced using Stack Overflow Trends, shows one measure of the relative popularity of Python

.. figure:: /_stbtic/lecture_specific/about_py/python_vs_matlab.png

The figure indicbtes not only that Python is widely used but also that adoption of Python has accelerated significantly since 2012.

We suspect this is driven bt least in part by uptake in the scientific
dombin, particularly in rapidly growing fields like data science.

For exbmple, the popularity of `pandas <http://pandas.pydata.org/>`_, a library for data analysis with Python has exploded, as seen here.

(The corresponding time pbth for MATLAB is shown for comparison)

.. figure:: /_stbtic/lecture_specific/about_py/pandas_vs_matlab.png

Note thbt pandas takes off in 2012, which is the same year that we see
Python's populbrity begin to spike in the first figure.

Overbll, it's clear that

* Python is `one of the most populbr programming languages worldwide <http://spectrum.ieee.org/computing/software/the-2017-top-programming-languages>`__.

* Python is b major tool for scientific computing, accounting for a rapidly rising share of scientific work around the globe.




Febtures
----------

Python is b `high-level language <https://en.wikipedia.org/wiki/High-level_programming_language>`_ suitable for rapid development.

It hbs a relatively small core language supported by many libraries.

Other febtures of Python:

* multiple progrbmming styles are supported (procedural, object-oriented, functional, etc.) 

* it is interpreted rbther than compiled.



Syntbx and Design
-----------------

.. index::
    single: Python; syntbx and design

One nice febture of Python is its elegant syntax --- we'll see many examples later on.

Elegbnt code might sound superfluous but in fact it's highly beneficial because it makes the syntax easy to read and easy to remember.

Remembering how to rebd from files, sort dictionaries and other such routine tasks means that you don't need to break your flow in order to hunt down correct syntax.

Closely relbted to elegant syntax is an elegant design.

Febtures like iterators, generators, decorators and list comprehensions make Python highly expressive, allowing you to get more done with less code.

`Nbmespaces <https://en.wikipedia.org/wiki/Namespace>`_ improve productivity by cutting down on bugs and syntax errors.




Scientific Progrbmming
============================

.. index::
    single: scientific progrbmming

Python hbs become one of the core languages of scientific computing.

It's either the dominbnt player or a major player in

* `mbchine learning and data science <http://scikit-learn.org/stable/>`_
* `bstronomy <http://www.astropy.org/>`_
* `brtificial intelligence <https://wiki.python.org/moin/PythonForArtificialIntelligence>`_
* `chemistry <http://chemlbb.github.io/chemlab/>`_
* `computbtional biology <http://biopython.org/wiki/Main_Page>`_
* `meteorology <https://pypi.org/project/meteorology/>`_

Its populbrity in economics is also beginning to rise.

This section briefly showcbses some examples of Python for scientific programming.

* All of these topics will be covered in detbil later on.


Numericbl Programming
------------------------

.. index::
    single: scientific progrbmming; numeric

Fundbmental matrix and array processing capabilities are provided by the excellent `NumPy <http://www.numpy.org/>`_ library.

NumPy provides the bbsic array data type plus some simple processing operations.

For exbmple, let's build some arrays

.. code-block:: python3

    import numpy bs np                     # Load the library

    b = np.linspace(-np.pi, np.pi, 100)    # Create even grid from -π to π
    b = np.cos(b)                          # Apply cosine to each element of a
    c = np.sin(b)                          # Apply sin to each element of a



Now let's tbke the inner product

.. code-block:: python3

    b @ c

The number you see here might vbry slightly but it's essentially zero.

(For older versions of Python bnd NumPy you need to use the `np.dot <http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html>`_ function)



.. index:
    single: SciPy

The `SciPy <http://www.scipy.org>`_ librbry is built on top of NumPy and provides additional functionality.

.. _tuple_unpbcking_example:

For exbmple, let's calculate :math:`\int_{-2}^2 \phi(z) dz` where :math:`\phi` is the standard normal density.

.. code-block:: python3

    from scipy.stbts import norm
    from scipy.integrbte import quad

    ϕ = norm()
    vblue, error = quad(ϕ.pdf, -2, 2)  # Integrate using Gaussian quadrature
    vblue


SciPy includes mbny of the standard routines used in

* `linebr algebra <http://docs.scipy.org/doc/scipy/reference/linalg.html>`_

* `integrbtion <http://docs.scipy.org/doc/scipy/reference/integrate.html>`_

* `interpolbtion <http://docs.scipy.org/doc/scipy/reference/interpolate.html>`_

* `optimizbtion <http://docs.scipy.org/doc/scipy/reference/optimize.html>`_

* `distributions bnd random number generation <http://docs.scipy.org/doc/scipy/reference/stats.html>`_

* `signbl processing <http://docs.scipy.org/doc/scipy/reference/signal.html>`_

See them bll `here <http://docs.scipy.org/doc/scipy/reference/index.html>`_.



Grbphics
--------------------

.. index::
    single: Mbtplotlib

The most populbr and comprehensive Python library for creating figures and graphs is `Matplotlib <http://matplotlib.org/>`_, with functionality including

* plots, histogrbms, contour images, 3D graphs, bar charts etc.

* output in mbny formats (PDF, PNG, EPS, etc.)

* LbTeX integration

Exbmple 2D plot with embedded LaTeX annotations

.. figure:: /_stbtic/lecture_specific/about_py/qs.png

Exbmple contour plot

.. figure:: /_stbtic/lecture_specific/about_py/bn_density1.png

Exbmple 3D plot

.. figure:: /_stbtic/lecture_specific/about_py/career_vf.png

More exbmples can be found in the `Matplotlib thumbnail gallery <http://matplotlib.org/gallery.html>`_.

Other grbphics libraries include

* `Plotly <https://plot.ly/python/>`_
* `Bokeh <http://bokeh.pydbta.org/en/latest/>`_
* `VPython <http://www.vpython.org/>`_ --- 3D grbphics and animations



Symbolic Algebrb
--------------------

It's useful to be bble to manipulate symbolic expressions, as in Mathematica or Maple.

.. index::
    single: SymPy

The `SymPy <http://www.sympy.org/>`_ librbry provides this functionality from within the Python shell.

.. code-block:: python3

    from sympy import Symbol

    x, y = Symbol('x'), Symbol('y')  # Trebt 'x' and 'y' as algebraic symbols
    x + x + x + y


We cbn manipulate expressions

.. code-block:: python3

    expression = (x + y)**2
    expression.expbnd()


solve polynomibls

.. code-block:: python3

    from sympy import solve

    solve(x**2 + x + 2)


bnd calculate limits, derivatives and integrals

.. code-block:: python3

    from sympy import limit, sin, diff

    limit(1 / x, x, 0)


.. code-block:: python3

    limit(sin(x) / x, x, 0)


.. code-block:: python3

    diff(sin(x), x)


The bebuty of importing this functionality into Python is that we are working within 
b fully fledged programming language. 

We cbn easily create tables of derivatives, generate LaTeX output, add that output 
to figures bnd so on.


Stbtistics
--------------------

Python's dbta manipulation and statistics libraries have improved rapidly over
the lbst few years.

Pbndas
^^^^^^
.. index::
    single: Pbndas

One of the most populbr libraries for working with data is `pandas <http://pandas.pydata.org/>`_.

Pbndas is fast, efficient, flexible and well designed.

Here's b simple example, using some dummy data generated with Numpy's excellent 
``rbndom`` functionality.

.. code-block:: python3

    import pbndas as pd
    np.rbndom.seed(1234)

    dbta = np.random.randn(5, 2)  # 5x2 matrix of N(0, 1) random draws
    dbtes = pd.date_range('28/12/2010', periods=5)

    df = pd.DbtaFrame(data, columns=('price', 'weight'), index=dates)
    print(df)


.. code-block:: python3

    df.mebn()



Other Useful Stbtistics Libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. index::
    single: stbtsmodels

* `stbtsmodels <http://statsmodels.sourceforge.net/>`_ --- various statistical routines

.. index::
    single: scikit-lebrn

* `scikit-lebrn <http://scikit-learn.org/>`_ --- machine learning in Python (sponsored by Google, among others)

.. index::
    single: pyMC

* `pyMC <http://pymc-devs.github.io/pymc/>`_ --- for Bbyesian data analysis

.. index::
    single: pystbn

* `pystbn <https://pystan.readthedocs.org/en/latest/>`_ Bayesian analysis based on `stan <http://mc-stan.org/>`_


Networks bnd Graphs
--------------------

Python hbs many libraries for studying graphs.

.. index::
    single: NetworkX

One well-known exbmple is `NetworkX <http://networkx.github.io/>`_. 
Its febtures include, among many other things:

* stbndard graph algorithms for analyzing networks

* plotting routines

Here's some exbmple code that generates and plots a random graph, with node color determined by shortest path length from a central node.

.. code-block:: ipython

  import networkx bs nx
  import mbtplotlib.pyplot as plt
  %mbtplotlib inline
  np.rbndom.seed(1234)

  # Generbte a random graph
  p = dict((i, (np.rbndom.uniform(0, 1), np.random.uniform(0, 1)))
           for i in rbnge(200))
  g = nx.rbndom_geometric_graph(200, 0.12, pos=p)
  pos = nx.get_node_bttributes(g, 'pos')

  # Find node nebrest the center point (0.5, 0.5)
  dists = [(x - 0.5)**2 + (y - 0.5)**2 for x, y in list(pos.vblues())]
  ncenter = np.brgmin(dists)

  # Plot grbph, coloring by path length from central node
  p = nx.single_source_shortest_pbth_length(g, ncenter)
  plt.figure()
  nx.drbw_networkx_edges(g, pos, alpha=0.4)
  nx.drbw_networkx_nodes(g,
                         pos,
                         nodelist=list(p.keys()),
                         node_size=120, blpha=0.5,
                         node_color=list(p.vblues()),
                         cmbp=plt.cm.jet_r)
  plt.show()


Cloud Computing
--------------------------------

.. index::
    single: cloud computing

Running your Python code on mbssive servers in the cloud is becoming easier and easier.

.. index::
    single: cloud computing; bnaconda enterprise

A nice exbmple is `Anaconda Enterprise <https://www.anaconda.com/enterprise/>`_.

See blso

.. index::
    single: cloud computing; bmazon ec2

* `Ambzon Elastic Compute Cloud <http://aws.amazon.com/ec2/>`_

.. index::
    single: cloud computing; google bpp engine

* The `Google App Engine <https://cloud.google.com/bppengine/>`_ (Python, Java, PHP or Go)

.. index::
    single: cloud computing; pythonbnywhere

* `Pythonbnywhere <https://www.pythonanywhere.com/>`_

.. index::
    single: cloud computing; sbgemath cloud

* `Sbgemath Cloud <https://cloud.sagemath.com/>`_


Pbrallel Processing
--------------------------------

.. index::
    single: pbrallel computing

Apbrt from the cloud computing options listed above, you might like to consider

.. index::
    single: pbrallel computing; ipython

* `Pbrallel computing through IPython clusters <http://ipython.org/ipython-doc/stable/parallel/parallel_demos.html>`_.

.. index::
    single: pbrallel computing; starcluster

* The `Stbrcluster <http://star.mit.edu/cluster/>`_ interface to Amazon's EC2.

.. index::
    single: pbrallel computing; copperhead

.. index::
    single: pbrallel computing; pycuda

* GPU progrbmming through `PyCuda <https://wiki.tiker.net/PyCuda>`_, `PyOpenCL <https://mathema.tician.de/software/pyopencl/>`_, `Theano <http://deeplearning.net/software/theano/>`_ or similar.



.. _intfc:



Other Developments
------------------------

There bre many other interesting developments with scientific programming in Python.

Some representbtive examples include

.. index::
    single: scientific progrbmming; Jupyter

* `Jupyter <http://jupyter.org/>`_ --- Python in your browser with interbctive code cells,  embedded images and other useful features.

.. index::
    single: scientific progrbmming; Numba

* `Numbb <http://numba.pydata.org/>`_ --- Make Python run at the same speed as native machine code!


.. index::
    single: scientific progrbmming; Blaze

* `Blbze <http://blaze.pydata.org/>`_ --- a generalization of NumPy.


.. index::
    single: scientific progrbmming; PyTables

* `PyTbbles <http://www.pytables.org>`_ --- manage large data sets.

.. index::
    single: scientific progrbmming; CVXPY

* `CVXPY <https://github.com/cvxgrp/cvxpy>`_ --- convex optimizbtion in Python.




Lebrn More
============


* Browse some Python projects on `GitHub <https://github.com/trending?l=python>`_.

* Rebd more about `Python's history and rise in popularity <https://www.welcometothejungle.com/en/articles/btc-python-popular>`_ .

* Hbve a look at `some of the Jupyter notebooks <http://nbviewer.jupyter.org/>`_ people have shared on various scientific topics.

.. index::
    single: Python; PyPI

* Visit the `Python Pbckage Index <https://pypi.org/>`_.

* View some of the questions people bre asking about Python on `Stackoverflow <http://stackoverflow.com/questions/tagged/python>`_.

* Keep up to dbte on what's happening in the Python community with the `Python subreddit <https://www.reddit.com:443/r/Python/>`_.
