.. Emacs, this is -*-rst-*-
.. highlight:: rst

Gridworlds
==========

The ``gridworld`` module provides routines for working with
2-dimensional, 4-connected rectangular grids. Examples are given in
the directory ``examples/gridworlds``.  The primary source of detailed
documentation is in `docstrings
<http://www.python.org/dev/peps/pep-0257/#what-is-a-docstring>`_ found
in the source code itself. There are several ways to get at this;
e.g., you could use `pydoc <http://docs.python.org/library/pydoc.html>`_ by

.. highlight:: none
::

  $ pydoc tulip.gridworld

(See :ref:`venv-pydoc-sec-label` for troubleshooting.)  The basic
representation is provided by the class ``GridWorld``.  In all example
code below, we assume the ``gridworld`` module has been imported as
``gw``

.. highlight:: python
::

  import tulip.gridworld as gw


Description format
------------------

A gridworld problem can be defined by a "gridworld description
string." The core parsing routine is the ``GridWorld.loads``; it is
the current reference implementation.

.. automethod:: gridworld.GridWorld.loads

Examples
````````

Consider a 2 x 3 gridworld where you wish to declare the cell at (1,2)
(the bottom-right) as the initial position and (1,0) as a goal
cell. This is achieved with the description string

.. highlight:: none
::

  #    0 1 2
  #   -------
  #  0| |*| |
  #   -------
  #  1|G| |I|
  #   -------

  2 3
   *
  G I

Any line beginning with ``#`` is treated as a comment and ignored. The
purpose of the comment in this example is to provide an annotated view
of the grid; for large problems, comments can be very helpful for
humans, and anyway provide a means to make notes such as who created
the file, a timestamp, and other experimental parameters. In this
example, only the last three lines are critical. ``2 3`` declares that
there are 2 rows and 3 columns. The other two lines define grid
contents. Explicitly, there is a static obstacle at the first row and
second column (i.e., at (0,1)). The last line indicates where the goal
``G`` and initial position ``I`` should be.

.. highlight:: python

If the above description string were in a file called ``trivial.txt``,
then you could load it using ::

  with open("trivial.txt", "r") as f:
      triv = gw.GridWorld(f.read(), prefix="Y")

To prettily print the result, and then to print the variable name of
the cell located at (0,0), you would then ::

  print(triv)
  print(triv[0,0])

See the method ``pretty`` for more formatting options (the first line
above internally invokes ``pretty`` with sane defaults).  Notice that
the variable name has prefix "Y". This could be changed in the
``prefix`` argument used above when instantiating ``GridWorld``. The
string returned by ``triv[0,0]`` can be written in specifications.
Indexing follows that of Python; in particular, negative indices are
supported.


Generating continuous-space partitions
--------------------------------------

Given a ``GridWorld`` object ``Y``, you can create a
``PropPreservingPartition`` object describing the grid in a continuous
state space with the method ``dump_ppartition``.
An example is to generate a random gridworld, generate an initial
proposition-preserving partition, and then refine it based on
continuous state space dynamics, as shown in the code below. Note that
we use mostly default argument values to minimize clutter.

.. highlight:: python
::

  import numpy as np
  from tulip.abstract import discretize
  from tulip import gridworld as gw
  from tulip.hybrid import LtiSysDyn
  from polytope import Polytope
  from polytope.plot import plot_partition

  # Trivial dynamics
  A = np.eye(2)
  B = np.eye(2)
  E = np.eye(2)
  U = Polytope(np.array([[1., 0.],[-1., 0.], [0., 1.], [0., -1.]]),
	       np.array([[1.],[1.],[1.],[1.]]))
  W = Polytope(np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]]),
	       0.01*np.array([1., 1., 1., 1.]))
  sys_dyn = LtiSysDyn(A,B,E, Uset=U, Wset=W)

  # Generate random gridworld, dump it and discretize based on dynamics
  Y = gw.random_world((5, 10), num_init=0, num_goals=0)
  disc_dynamics = discretize(Y.dump_ppartition(), sys_dyn)

  # Pretty print abstraction to terminal, and depict partition reachability
  print(Y)
  plot_partition(disc_dynamics.ppp, trans=True)
