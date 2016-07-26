Introduction
============

The `Temporal Logic Planning (TuLiP) Toolbox
<http://tulip-control.org>`_ is a collection of Python-based code for
automatic synthesis of correct-by-construction embedded control software as
discussed in `[FDLOM16] <bibliography.html#fdlom16>`_.  This chapter contains a brief overview of the toolbox,
along with instructions on how to install the software.

Though installation should be easy (and standardized) through setup.py, there
are some dependencies that will need to be manually installed if not already
available; details are provided in :doc:`install`.  Note that *this
documentation is still under development*.

Package Overview
----------------

TuLiP is designed to synthesize discrete-state controllers for hybrid systems
operating in a (potentially dynamic and unknown) environment.  The system
specification is given in terms of a temporal logic formula, typically written
in LTL.

The approach used by TuLiP is outlined in the figure below:

.. image:: approach.png
   :alt: Block diagram of the flow from a system model and specification to
         continuous and discrete parts of a constructed controller. Between
         these inputs and outputs, several blocks are grouped together
         indicating various kinds of routines and objects in TuLiP:
         proposition-preserving partition, continuous state space
         discretization, finite transition system, and synthesis.

The procedure that we used is broken down into three primary steps:

  * Construct a finite transition system :math:`D` (e.g. a Kripke structure)
    that serves as an abstract model of :math:`S` (which typically has
    infinitely many states)

  * Synthesize a discrete planner that computes a discrete plan satisfying
    the specification :math:`\varphi` based on the abstract, finite-state
    model :math:`D`;

  * Design a continuous controller that implements the discrete plan.

More information on the solution strategy is available in `[FDLOM16] <bibliography.html#fdlom16>`_ and
:doc:`formulations`.

Version 1.0 Release Notes
-------------------------
Version 1.0 of TuLiP represents a major overhaul of the structure of the
code to allow better support for integration with other tools and adding
functionality.  Code and examples for version 0.x of TuLiP are not
compatible with version 1.0+ and must be rewritten from scratch.

Other sources of documentation
------------------------------

You are currently reading the User's Guide.  There is also API documentation,
which provides details about the various classes, methods, etc. in TuLiP.  This
can be accessed using the standard `pydoc
<https://docs.python.org/2.7/library/pydoc.html>`_ tool.  E.g., ::

  pydoc tulip

The API documentation is also available through a richer interface that
includes, among other things, hyperlinks and inheritance diagrams.  It is
generated using `Epydoc <http://epydoc.sourceforge.net/>`_ and can be built from
the ``doc`` directory in the TuLiP sources::

  make api

Built copies for the most recent release of TuLiP are available online at:

* http://tulip-control.sourceforge.net/doc/
* http://tulip-control.sourceforge.net/api-doc/

Getting help
------------

* Visit the `#tulip-control <https://webchat.freenode.net/?channels=tulip-control>`_ channel of the `freenode <http://freenode.net/>`_ IRC network.
* Contact project members at tulip@tulip-control.org.
* Possible bug reports and feature requests can be made by `opening an issue <https://github.com/tulip-control/tulip-control/issues>`_ on `the project site at GitHub <https://github.com/tulip-control/tulip-control/>`_
