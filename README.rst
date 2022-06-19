TuLiP
=====
This is the source repository for TuLiP, the temporal logic planning toolbox.
The project website is http://tulip-control.org

Installation
------------

In most cases, it suffices to::

  pip install .

TuLiP can be installed also `from PyPI <https://pypi.python.org/pypi/tulip>`_::

  pip install tulip

This will install the latest release, together with required dependencies.
To find out what dependencies (including optional ones) are installed, call::

  tulip.interfaces.print_env()

For detailed instructions, including notes about dependencies and troubleshooting,
consult https://tulip-control.sourceforge.io/doc/install.html
The next section describes how to build documentation.
A test suite is provided under ``tests/``.  Consult the section "Testing" below.

Pip can install the latest *development* snapshot too::

  pip install https://github.com/tulip-control/tulip-control/archive/master.zip

Code under development can be unstable so trying `pip install tulip` first
is recommended.


Documentation
-------------

There are two main sources of documentation outside the code.  The "user"
documentation is under ``doc/`` and is built with `Sphinx
<https://www.sphinx-doc.org/>`_, so the usual steps apply, ::

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

Tests are performed using `pytest <https://pytest.org>`_.  From the
root of the source tree (i.e., where ``setup.py`` is located), ::

  ./run_tests.py

to perform basic tests.  To try all available tests, ``./run_tests.py full``.
For alternatives and a summary of usage, ``./run_tests.py -h``


Contact and support
-------------------

* Ask for help on the `tulip-control-users mailing list <https://sourceforge.net/p/tulip-control/mailman/tulip-control-users>`_
* For release announcements, join the `tulip-control-announce mailing list <https://sourceforge.net/p/tulip-control/mailman/tulip-control-announce>`_
* Bug reports and feature requests should be made at https://github.com/tulip-control/tulip-control/issues
  Please check for prior discussion and reports before opening a new issue.


License
-------

This is free software released under the terms of `the BSD 3-Clause License
<https://opensource.org/licenses/BSD-3-Clause>`_.  There is no warranty; not even
for merchantability or fitness for a particular purpose.  Consult LICENSE for
copying conditions.

When code is modified or re-distributed, the LICENSE file should accompany the code or any subset of it, however small.
As an alternative, the LICENSE text can be copied within files, if so desired.
