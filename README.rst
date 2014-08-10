TuLiP
=====
This is the source repository for TuLiP, the temporal logic planning toolbox.
The project website is http://tulip-control.org

Installation
------------

In most cases, it suffices to::

  python setup.py install

To avoid checking for optional dependencies, add the option "nocheck"::

  python setup.py install nocheck

Detailed instructions, including notes about dependencies and troubleshooting,
are available at http://tulip-control.sourceforge.net/doc/install.html

The next section describes how to build documentation.  A test suite is provided
under ``tests/``.  Consult the section "Testing" below.


Documentation
-------------

There are two main sources of documentation outside the code.  The "user"
documentation is under ``doc/`` and is built with `Sphinx
<http://sphinx.pocoo.org/>`_, so the usual steps apply, ::

  make html

API documentation is generated using `Epydoc <http://epydoc.sourceforge.net/>`_
and can also be built from the ``doc`` directory, now by ::

  make api

Built copies for the most recent release of TuLiP are available online at:

* http://tulip-control.sourceforge.net/doc/
* http://tulip-control.sourceforge.net/api-doc/

Command summaries are provided by ``make help``.  Besides the above sources, you
may also read API documentation using the standard pydoc tool.  E.g., ::

  pydoc tulip


Testing
-------

Tests are performed using nose; see http://readthedocs.org/docs/nose/ for
details.  From the root of the source tree (i.e., where setup.py is located),
run::

  ./run_tests.py

to run all available tests.  Use the flag "-h" to see driver script options.  To
disable output capture, add the flag "-s" when invoking nose.
