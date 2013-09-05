Introduction
============

The Temporal Logic Planning (TuLiP) Toolbox is a collection of Python-based
code for automatic synthesis of correct-by-construction embedded control
software as discussed in [1]_, [2]_.  This chapter contains a brief overview
of the toolbox, along with instructions on how to install the software.

Please also see `the wiki <http://tulip-control.sourceforge.net>`_ for
additional information and related links.  Though installation should be
easy (and standardized) through setup.py, there are some dependencies that
will need to be manually installed if not already available; details are
provided in :doc:`install`.  Note that *this documentation is still under
development*.  For help and support, please visit `our support page
<http://sourceforge.net/projects/tulip-control/support>`_.

Package Overview
----------------

TuLiP is designed to solve synthesize discrete-state controllers for hybrid
systems operating in a (potentially dynamic and unknown) environment.  The
system specification is given in terms of a temporal logic formula,
typically written in LTL.

The approach used by TuLiP is outlined in the figure below:

.. image:: approach.*

The procedure that we used is broken down into three primary steps:

  * Construct a finite transition system :math:`D` (e.g. a Kripke structure)
    that serves as an abstract model of :math:`S` (which typically has
    infinitely many states)

  * Synthesize a discrete planner that computes a discrete plan satisfying
    the specification :math:`\varphi` based on the abstract, finite-state
    model :math:`D`;

  * Design a continuous controller that implements the discrete plan.

More information on the solution strategy is available in [1].

.. include:: install.txt

Version 1.0 Release Notes
-------------------------
Version 1.0 of TuLiP represents a major overhaul of the structure of the
code to allow better support for integration with other tools and adding
functionality.  Code and examples for version 0.x of TuLiP are not
compatible with version 1.0+ and must be rewritten from scratch.

References
----------

.. [1] `TuLiP: a software toolbox for receding horizon
  temporal logic planning <http://www.cds.caltech.edu/~utopcu/images//d/d2/WTOXM-hscc11.pdf>`_, T. Wongpiromsarn, U. Topcu, N. Ozay, H. Xu,
  and R. Murray, International Conference on Hybrid Systems:
  Computation and Control, 2011.

.. [2] `Receding horizon temporal logic planning <http://www.cds.caltech.edu/~utopcu/images/0/09/WTM-itac10.pdf>`_,
  T. Wongpiromsarn, U. Topcu, and R. Murray, IEEE
  Transactions on Automatic Control, 57(11):2817-2830, 2012.
