TuLiP
=====

.. image:: https://badges.gitter.im/tulip-control/tulip.svg
   :alt: Join the chat at https://gitter.im/tulip-control/tulip
   :target: https://gitter.im/tulip-control/tulip?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
This is the source repository for TuLiP, the temporal logic planning toolbox.
The project website is http://tulip-control.org

Installation
------------

In most cases, it suffices to::

  pip install .

TuLiP can be installed also `from PyPI <https://pypi.python.org/pypi/tulip>`_::

  pip install tulip

This will install the required dependencies.
To find out what dependencies (including optional ones) are installed, call::

  tulip.interfaces.print_env()

For detailed instructions, including notes about dependencies and troubleshooting,
consult https://tulip-control.sourceforge.io/doc/install.html
The next section describes how to build documentation.
A test suite is provided under ``tests/``.  Consult the section "Testing" below.


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

* https://tulip-control.sourceforge.io/doc/
* https://tulip-control.sourceforge.io/api-doc/

Command summaries are provided by ``make help``.  Besides the above sources, you
may also read API documentation using the standard pydoc tool.  E.g., ::

  pydoc tulip


Testing
-------

Tests are performed using `nose <http://readthedocs.org/docs/nose/>`_.  From the
root of the source tree (i.e., where ``setup.py`` is located), ::

  ./run_tests.py

to perform basic tests.  To try all available tests, ``./run_tests.py full``.
For alternatives and a summary of usage, ``./run_tests.py -h``


License
-------

This is free software released under the terms of `the BSD 3-Clause License
<http://opensource.org/licenses/BSD-3-Clause>`_.  There is no warranty; not even
for merchantability or fitness for a particular purpose.  Consult LICENSE for
copying conditions.

When code is modified or re-distributed, the LICENSE file should accompany the code or any subset of it, however small.
As an alternative, the LICENSE text can be copied within files, if so desired.
