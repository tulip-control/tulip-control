TuLiP
=====
This is the source repository for TuLiP, the temporal logic planning toolbox.

Installation
------------

In most cases, it suffices to::

  python setup.py install

To avoid checking for required, recommended, and optional dependencies, add the
option "nocheck"::

  python setup.py install nocheck

Detailed instructions, including notes about dependencies and troubleshooting,
are available at

  http://tulip-control.sourceforge.net/doc/install.html

The documentation sources (see below) can be found under ``doc/``.  A test suite
(see below) is provided under tests/.


Sphinx and Epydoc generated documentation
-----------------------------------------

There are two main sources of documentation outside the code.  The "user"
documentation is under ``doc/`` and is built with `Sphinx
<http://sphinx.pocoo.org/>`_, so the usual steps apply, ::

  make html

Note that a Windows build file, make.bat, was auto-generated at the time of
first initialising the docs configuration files (ca. 2011 May
8) but is not actively maintained.  It is included for convenience; please
consider ``doc/Makefile`` to be the ground truth.  A built copy for the most
recent release of TuLiP is available online at

  http://tulip-control.sourceforge.net/doc/

API documentation is generated using `Epydoc <http://epydoc.sourceforge.net/>`_
and can also be built from the ``doc`` directory, now by ::

  make api

Command summaries are provided by ``make help``.


Testing
-------

Tests are performed using nose; see http://readthedocs.org/docs/nose/ for
details.  From the root of the source tree (i.e., where setup.py is located),
run::

  ./run_tests.py

to run all available tests.  Use the flag "-h" to see driver script options.  To
change default options, edit the "nosetests" section in setup.cfg.  To disable
output capture, add the flag "-s" when invoking nose.
