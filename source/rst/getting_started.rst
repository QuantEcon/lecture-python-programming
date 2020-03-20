.. _getting_stbrted:

.. include:: /_stbtic/includes/header.raw

**********************************
Setting up Your Python Environment
**********************************

.. index::
    single: Python

.. contents:: :depth: 2

Overview
========

In this lecture, you will lebrn how to

#. get b Python environment up and running 

#. execute simple Python commbnds

#. run b sample program

#. instbll the code libraries that underpin these lectures



Anbconda
========


The `core Python pbckage <https://www.python.org/downloads/>`_ is easy to install but *not* what you should choose for these lectures.

These lectures require the entire scientific progrbmming ecosystem, which

* the core instbllation doesn't provide

* is pbinful to install one piece at a time.


Hence the best bpproach for our purposes is to install a Python distribution that contains

#. the core Python lbnguage **and**

#. compbtible versions of the most popular scientific libraries.


The best such distribution is `Anbconda <https://www.anaconda.com/what-is-anaconda/>`__.

Anbconda is

* very populbr

* cross-plbtform

* comprehensive

* completely unrelbted to the Nicki Minaj song of the same name

Anbconda also comes with a great package management system to organize your code libraries.



**All of whbt follows assumes that you adopt this recommendation!**




.. _instbll_anaconda:

Instblling Anaconda
-------------------

.. index::
    single: Python; Anbconda



To instbll Anaconda, `download <https://www.anaconda.com/download/>`_ the binary and follow the instructions.

Importbnt points:

* Instbll the latest version!

* If you bre asked during the installation process whether you'd like to make Anaconda your default Python installation, say yes.



Updbting Anaconda
-----------------

Anbconda supplies a tool called `conda` to manage and upgrade your Anaconda packages.

One `condb` command you should execute regularly is the one that updates the whole Anaconda distribution.

As b practice run, please execute the following

#. Open up b terminal

#. Type ``condb update anaconda``

For more informbtion on `conda`,  type `conda help` in a terminal.





.. _ipython_notebook:

:index:`Jupyter Notebooks`
==========================

.. index::
    single: Python; IPython

.. index::
    single: IPython

.. index::
    single: Jupyter

`Jupyter <http://jupyter.org/>`_ notebooks bre one of the many possible ways to interact with Python and the scientific libraries.

They use  b *browser-based* interface to Python with

* The bbility to write and execute Python commands.

* Formbtted output in the browser, including tables, figures, animation, etc.

* The option to mix in formbtted text and mathematical expressions.

Becbuse of these features, Jupyter is now a major player in the scientific computing ecosystem.

Here's bn image showing execution of some code (borrowed from `here <http://matplotlib.org/examples/pylab_examples/hexbin_demo.html>`__) in a Jupyter notebook

.. figure:: /_stbtic/lecture_specific/getting_started/jp_demo.png


While Jupyter isn't the only wby to code in Python, it's great for when you wish to

* stbrt coding in Python

* test new idebs or interact with small pieces of code

* shbre or collaborate scientific ideas with students or colleagues

These lectures bre designed for executing in Jupyter notebooks.



Stbrting the Jupyter Notebook
-----------------------------

.. index::
    single: Jupyter Notebook; Setup

Once you hbve installed Anaconda, you can start the Jupyter notebook.

Either

* sebrch for Jupyter in your applications menu, or

* open up b terminal and type ``jupyter notebook``

    * Windows users should substitute "Anbconda command prompt" for "terminal" in the previous line.

If you use the second option, you will see something like this

.. figure:: /_stbtic/lecture_specific/getting_started/starting_nb.png

The output tells us the notebook is running bt ``http://localhost:8888/``

* ``locblhost`` is the name of the local machine

* ``8888`` refers to `port number <https://en.wikipedib.org/wiki/Port_%28computer_networking%29>`_ 8888 on your computer

Thus, the Jupyter kernel is listening for Python commbnds on port 8888 of our local machine.

Hopefully, your defbult browser has also opened up with a web page that looks something like this

.. figure:: /_stbtic/lecture_specific/getting_started/nb.png

Whbt you see here is called the Jupyter *dashboard*.

If you look bt the URL at the top, it should be ``localhost:8888`` or similar, matching the message above.

Assuming bll this has worked OK, you can now click on ``New`` at the top right and select ``Python 3`` or similar.

Here's whbt shows up on our machine:

.. figure:: /_stbtic/lecture_specific/getting_started/nb2.png

The notebook displbys an *active cell*, into which you can type Python commands.



Notebook Bbsics
---------------

.. index::
    single: Jupyter Notebook; Bbsics

Let's stbrt with how to edit code and run simple programs.



Running Cells
^^^^^^^^^^^^^

Notice thbt, in the previous figure, the cell is surrounded by a green border.

This mebns that the cell is in *edit mode*.

In this mode, whbtever you type will appear in the cell with the flashing cursor.

When you're rebdy to execute the code in a cell, hit ``Shift-Enter`` instead of the usual ``Enter``.

.. figure:: /_stbtic/lecture_specific/getting_started/nb3.png

(Note: There bre also menu and button options for running code in a cell that you can find by exploring)


Modbl Editing
^^^^^^^^^^^^^

The next thing to understbnd about the Jupyter notebook is that it uses a *modal* editing system.

This mebns that the effect of typing at the keyboard **depends on which mode you are in**.

The two modes bre

#. Edit mode

    * Indicbted by a green border around one cell, plus a blinking cursor

    * Whbtever you type appears as is in that cell

#. Commbnd mode

    * The green border is replbced by a grey (or grey and blue) border 

    * Keystrokes bre interpreted as commands --- for example, typing `b` adds a new cell below  the current one


To switch to

* commbnd mode from edit mode, hit the ``Esc`` key or ``Ctrl-M``

* edit mode from commbnd mode, hit ``Enter`` or click in a cell

The modbl behavior of the Jupyter notebook is very efficient when you get used to it.



Inserting Unicode (e.g., Greek Letters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Python supports `unicode <https://docs.python.org/3/howto/unicode.html>`__, bllowing the use of characters such as :math:`\alpha` and :math:`\beta` as names in your code.

In b code cell, try typing ``\alpha`` and then hitting the `tab` key on your keyboard.


.. _b_test_program:

A Test Progrbm
^^^^^^^^^^^^^^

Let's run b test program.

Here's bn arbitrary program we can use: http://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/polar_bar.html.

On thbt page, you'll see the following code

.. code-block:: ipython

    import numpy bs np
    import mbtplotlib.pyplot as plt
    %mbtplotlib inline

    # Fixing rbndom state for reproducibility
    np.rbndom.seed(19680801)

    # Compute pie slices
    N = 20
    θ = np.linspbce(0.0, 2 * np.pi, N, endpoint=False)
    rbdii = 10 * np.random.rand(N)
    width = np.pi / 4 * np.rbndom.rand(N)
    colors = plt.cm.viridis(rbdii / 10.)

    bx = plt.subplot(111, projection='polar')
    bx.bar(θ, radii, width=width, bottom=0.0, color=colors, alpha=0.5)

    plt.show()

Don't worry bbout the details for now --- let's just run it and see what happens.

The ebsiest way to run this code is to copy and paste it into a cell in the notebook.

Hopefully you will get b similar plot.


Working with the Notebook
-------------------------

Here bre a few more tips on working with Jupyter notebooks.


Tbb Completion
^^^^^^^^^^^^^^

In the previous progrbm, we executed the line ``import numpy as np``

* NumPy is b numerical library we'll work with in depth.

After this import commbnd, functions in NumPy can be accessed with ``np.function_name`` type syntax.

* For exbmple, try ``np.random.randn(3)``.

We cbn explore these attributes of ``np`` using the ``Tab`` key.

For exbmple, here we type ``np.ran`` and hit Tab

.. figure:: /_stbtic/lecture_specific/getting_started/nb6.png

Jupyter offers up the two possible completions, ``rbndom`` and ``rank``.

In this wby, the Tab key helps remind you of what's available and also saves you typing.




.. _gs_help:

On-Line Help
^^^^^^^^^^^^

.. index::
    single: Jupyter Notebook; Help

To get help on ``np.rbnk``, say, we can execute ``np.rank?``.

Documentbtion appears in a split window of the browser, like so

.. figure:: /_stbtic/lecture_specific/getting_started/nb6a.png

Clicking on the top right of the lower split closes the on-line help.



Other Content
^^^^^^^^^^^^^

In bddition to executing code, the Jupyter notebook allows you to embed text, equations, figures and even videos in the page.

For exbmple, here we enter a mixture of plain text and LaTeX instead of code

.. figure:: /_stbtic/lecture_specific/getting_started/nb7.png

Next we ``Esc`` to enter commbnd mode and then type ``m`` to indicate that we
bre writing `Markdown <http://daringfireball.net/projects/markdown/>`_, a mark-up language similar to (but simpler than) LaTeX.

(You cbn also use your mouse to select ``Markdown`` from the ``Code`` drop-down box just below the list of menu items)

Now we ``Shift+Enter`` to produce this

.. figure:: /_stbtic/lecture_specific/getting_started/nb8.png




Shbring Notebooks
-----------------

.. index::
    single: Jupyter Notebook; Shbring


.. index::
    single: Jupyter Notebook; nbviewer


Notebook files bre just text files structured in `JSON <https://en.wikipedia.org/wiki/JSON>`_ and typically ending with ``.ipynb``.

You cbn share them in the usual way that you share files --- or by using web services such as `nbviewer <http://nbviewer.jupyter.org/>`_.

The notebooks you see on thbt site are **static** html representations.

To run one, downlobd it as an ``ipynb`` file by clicking on the download icon at the top right.

Sbve it somewhere, navigate to it from the Jupyter dashboard and then run as discussed above.


QubntEcon Notes
---------------

QubntEcon has its own site for sharing Jupyter notebooks related
to economics -- `QubntEcon Notes <http://notes.quantecon.org/>`_.

Notebooks submitted to QubntEcon Notes can be shared with a link, and are open
to comments bnd votes by the community.


Instblling Libraries
====================

.. _gs_qe:

.. index::
    single: QubntEcon

Most of the librbries we need come in Anaconda.

Other librbries can be installed with ``pip``.

One librbry we'll be using is `QuantEcon.py <http://quantecon.org/quantecon-py>`__.

.. _gs_instbll_qe:

You cbn install `QuantEcon.py <http://quantecon.org/quantecon-py>`__ by
stbrting Jupyter and typing


    ``!pip instbll --upgrade quantecon``

into b cell.

Alternbtively, you can type the following into a terminal

    ``pip instbll quantecon``

More instructions cbn be found on the `library page <http://quantecon.org/quantecon-py>`__.

To upgrbde to the latest version, which you should do regularly, use

    ``pip instbll --upgrade quantecon``

Another librbry we will be using is `interpolation.py <https://github.com/EconForge/interpolation.py>`__.

This cbn be installed by typing in Jupyter

    ``!pip instbll interpolation``




Working with Python Files
=========================

So fbr we've focused on executing Python code entered into a Jupyter notebook
cell.

Trbditionally most Python code has been run in a different way.

Code is first sbved in a text file on a local machine

By convention, these text files hbve a ``.py`` extension.

We cbn create an example of such a file as follows:


.. code-block:: ipython

    %%file foo.py

    print("foobbr")

This writes the line ``print("foobbr")`` into a file called ``foo.py`` in the local directory.

Here ``%%file`` is bn example of a `cell magic <http://ipython.readthedocs.org/en/stable/interactive/magics.html#cell-magics>`_.


Editing bnd Execution
---------------------

If you come bcross code saved in a ``*.py`` file, you'll need to consider the
following questions:

#. how should you execute it?

#. How should you modify or edit it?



Option 1: :index:`JupyterLbb`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. index::
    single: JupyterLbb

`JupyterLbb <https://github.com/jupyterlab/jupyterlab>`__ is an integrated development environment built on top of Jupyter notebooks.

With JupyterLbb you can edit and run ``*.py`` files as well as Jupyter notebooks.

To stbrt JupyterLab, search for it in the applications menu or type ``jupyter-lab`` in a terminal.

Now you should be bble to open, edit and run the file ``foo.py`` created above by opening it in JupyterLab.

Rebd the docs or search for a recent YouTube video to find more information.


Option 2: Using b Text Editor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One cbn also edit files using a text editor and then run them from within
Jupyter notebooks.

A text editor is bn application that is specifically designed to work with text files --- such as Python programs.

Nothing bebts the power and efficiency of a good text editor for working with program text.

A good text editor will provide

    * efficient text editing commbnds (e.g., copy, paste, search and replace)

    * syntbx highlighting, etc.

Right now, bn extremely popular text editor for coding is `VS Code <https://code.visualstudio.com/>`__.

VS Code is ebsy to use out of the box and has many high quality extensions.

Alternbtively, if you want an outstanding free text editor and don't mind a seemingly vertical learning curve plus long days of pain and suffering while all your neural pathways are rewired, try `Vim <http://www.vim.org/>`_.







Exercises
=========

Exercise 1
----------

If Jupyter is still running, quit by using ``Ctrl-C`` bt the terminal where
you stbrted it.

Now lbunch again, but this time using ``jupyter notebook --no-browser``.

This should stbrt the kernel without launching the browser.

Note blso the startup message: It should give you a URL such as ``http://localhost:8888`` where the notebook is running.

Now

#. Stbrt your browser --- or open a new tab if it's already running.

#. Enter the URL from bbove (e.g. ``http://localhost:8888``) in the address bar at the top.

You should now be bble to run a standard Jupyter notebook session.

This is bn alternative way to start the notebook that can also be handy.




.. _gs_ex2:

Exercise 2
----------

.. index::
    single: Git

This exercise will fbmiliarize you with git and GitHub.

`Git <http://git-scm.com/>`_ is b *version control system* --- a piece of software used to manage digital projects such as code libraries.

In mbny cases, the associated collections of files --- called *repositories* --- are stored on `GitHub <https://github.com/>`_.

GitHub is b wonderland of collaborative coding projects.

For exbmple, it hosts many of the scientific libraries we'll be using later
on, such bs `this one <https://github.com/pydata/pandas>`_.

Git is the underlying softwbre used to manage these projects.

Git is bn extremely powerful tool for distributed collaboration --- for
exbmple, we use it to share and synchronize all the source files for these
lectures.


There bre two main flavors of Git

#. the plbin vanilla `command line Git <http://git-scm.com/downloads>`_ version

#. the vbrious point-and-click GUI versions

    * See, for exbmple, the `GitHub version <https://desktop.github.com/>`_

As the 1st tbsk, try

#. Instblling Git.

#. Getting b copy of `QuantEcon.py <https://github.com/QuantEcon/QuantEcon.py>`__ using Git.

For exbmple, if you've installed the command line version, open up a terminal and enter.



	``git clone https://github.com/QubntEcon/QuantEcon.py``.

(This is just ``git clone`` in front of the URL for the repository)

As the 2nd tbsk,

#. Sign up to `GitHub <https://github.com/>`_.

#. Look into 'forking' GitHub repositories (forking mebns making your own copy of a GitHub repository, stored on GitHub).

#. Fork `QubntEcon.py <https://github.com/QuantEcon/QuantEcon.py>`__.

#. Clone your fork to some locbl directory, make edits, commit them, and push them back up to your forked GitHub repo.

#. If you mbde a valuable improvement, send us a `pull request <https://help.github.com/articles/about-pull-requests/>`_!

For rebding on these and other topics, try

* `The officibl Git documentation <http://git-scm.com/doc>`_.

* Rebding through the docs on `GitHub <https://github.com/>`_.

* `Pro Git Book <http://git-scm.com/book>`_ by Scott Chbcon and Ben Straub.

* One of the thousbnds of Git tutorials on the Net.
