Developer's Guide
=================

The purpose of this page is to provide guidelines for contributors to the TuLiP
project.  Also consult the `Developers' Wiki <https://github.com/tulip-control/tulip-control/wiki>`_ and the `tulip-control-discuss mailing list <https://sourceforge.net/p/tulip-control/mailman/tulip-control-discuss/>`_ (members only).

.. _sec:code-style-guidelines:

Organization and Rules
----------------------

We begin with important organizational notes and **rules** that should
be followed:

- `PEP 20 <https://www.python.org/dev/peps/pep-0020/>`_.

- `PEP 8 <http://python.org/dev/peps/pep-0008/>`_.  Especially, you should

  - use 4-space indentation;
  - keep lines as short as possible, and at most 79 characters;
  - name classes like ``OurExample``, and methods like ``generate_names``;
    notice that the former name is an object, whereas the latter is a command.

- Avoid trailing whitespace.

- `PEP 257 <http://python.org/dev/peps/pep-0257/>`_.  In summary, you should
  write a docstring for any public object (e.g., classes, methods, modules), and
  it should begin with a single-line summary.  This summary should be an
  imperative statement (e.g., "Compute the volume of given polytope.").  Any
  additional documentation beyond the single-line summary can be included after
  leaving a blank line.
- Be careful what you export, i.e., make sure that what is obtained when someone
  uses "from tulip.foo import \*" is what you intend.  Otherwise, hide names
  using the `"_" prefix <http://docs.python.org/2.7/reference/lexical_analysis.html#reserved-classes-of-identifiers>`_.
- API documentation is built using `Epydoc <http://epydoc.sourceforge.net/>`_.  Accordingly, docstrings should be marked up with `Epytext <http://epydoc.sourceforge.net/manual-epytext.html>`_.

- The User's and Developer's Guides are built using `Sphinx <http://sphinx.pocoo.org/>`_.  It uses a small extension of `reStructuredText <http://docutils.sourceforge.net/rst.html>`_.  Consult the `reST quick reference <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_.

- Besides the previous two sources, documentation can appear in plaintext files, notably in README files.  These should have line widths of at most 80 characters.  E.g., this can be achieved at the command-line using ``fold -s -w 80`` or in `Emacs <http://www.gnu.org/software/emacs>`_ by ``C-u 80 M-x set-fill-column``.

- When committing to the repository, you should write a summary line, at most 60
  characters in length, and if elaboration is necessary, then first skip a line
  (i.e., leave one blank) before beginning with details.

- A copyright notice and pointer to the ``LICENSE`` file of ``tulip`` shall be placed
  as comments at the top of each source file (unless no copyright applies).

- When referring to publications, check for a corresponding entry in
  ``doc/bib.txt`` and create one if needed. The syntax is described in
  ``genbib.py``. References in the Sphinx-built documentation are achieved by
  including a link, e.g., inline like ::

    `[WTOXM11] <bibliography.html#wtoxm11>`_

  which renders as `[WTOXM11] <bibliography.html#wtoxm11>`_.  References in docstrings (in the
  code) should be to the URL of the corresponding entry on the TuLiP website,
  using `Epydoc syntax <http://epydoc.sourceforge.net/manual-epytext.html>`_,
  e.g., ::

    U{[WTOXM11] <http://tulip-control.sourceforge.net/doc/bibliography.html#wtoxm11>}

Testing
-------

A script for running tests is ``run_test.py`` in the root of the source
tree. Without the ``-f`` or ``--testfiles`` switch, ``run_tests.py`` expects the
user to request a family of tests to perform. The default is "base", which
corresponds to tests that should pass when the required dependencies of TuLiP
are satisfied. The other extreme is "full", which performs all tests. In
between, other families are defined, e.g., "hybrid", which involves all "base"
tests and any tests that should pass given a successful ``pip install tulip[hybrid]``,
namely, when the optional packages ``cvxopt`` and ``polytope`` are present.

Provided the ``-f`` or ``--testfiles`` switch, it searches under the directory
``tests/`` for files with names ending in "_test.py", and passes these to `nose
<http://readthedocs.org/docs/nose/>`_.  Use the flag "-h" to see driver script
options.  Extra details about options:

* The flag "--cover" to generate a coverage report, which will likely be placed
  under ``tests/cover/``.  It uses `Ned Batchelder's coverage module
  <http://www.nedbatchelder.com/code/modules/coverage.html>`_.

* The flag "--outofsource" will cause ``tulip`` to be imported from outside the
  current directory.  This is useful for testing against the installed form of
  TuLiP.

Version naming
--------------

For version numbers, the style is N.m.x where

* N = major version; everything that worked before could break
* m = revision; most functions should work, but might need (minor) modifications
* x = minor revision; code that ran should continue running

So, if you go from version 1.2.2 to 1.2.3, then no interfaces should
change, code should continue to run, etc.  If you go from 1.2.3 to
1.3.0, then there might be changes in some arguments lists or other
small things, but previous functionality still in place (somewhow).
If you go from 1.3.0 to 2.0.0, then we can make whatever changes we
want.

None of these version numbers go in individual files, but
the version number is a label for the entire package.

Making releases
---------------

#. Collect list of major changes.
#. Update the changelog.
#. Tag with message of the form "REL: version 1.2.0".
#. Create source release, ``python setup.py sdist``.
#. Post it to PyPI and SourceForge.net.
#. Build and post User's Guide and API manual. Under the directory doc/, run ::

     ./rsync-web.sh USERNAME
     ./rsync-docs.sh USERNAME

   where ``USERNAME`` is your SourceForge.net handle.
#. Make announcement on `tulip-control-announce mailing list
   <https://lists.sourceforge.net/lists/listinfo/tulip-control-announce>`_,
   providing major website links and the summary of changes.
#. Bump version in the repository, in preparation for next release.


Advice
------

The following are software engineering best practices that you should try to
follow.  We mention them here for convenience of reference and to aid new
committers. Unlike :ref:`sec:code-style-guidelines`, this section can be
entirely ignored.

- Keep function length to a minimum.
	As mentioned `at this talk <http://www.infoq.com/presentations/Scrub-Spin>`_, `MSL <http://en.wikipedia.org/wiki/Mars_Science_Laboratory>`_ included the rule that no function should be longer than 75 lines of code.
	The Linux coding style guide is succinct
	   "The answer to that is that if you need more than 3 levels of indentation,
	   you're screwed anyway,
	   and should fix your program."
	For example, within any iteration, usually the iterated code block deserves its own function (or method).
	This changes context, helping to focus at each level individually.
	Things can also be named better, reusing names within the iteration w/o conflicts.
	Incidentally it also saves from long lines.
	Besides these, short functions are viewable w/o vertical scrolling.
	When debugging after months, the shorter the function, the faster it is loaded to working memory.

- Avoid complicated conditions for if statements and other expressions.
	Break them down into simpler ones. When possible write them in sequence (not nested), so that they are checked in an obvious order.
	This way a function returns when a condition is False, so the conjunction is implicit and easier to follow, one check at a time.

- Name things to minimize comments.
	Comments are useless if they attempt to explain what the code evidently does and can be harmful if they fail to do so and instead describe what it was intended to do, giving a false impression of correctness.

- Have (simple) static checking on.
	e.g. `Spyder <http://code.google.com/p/spyderlib/>`_ with `pyflakes <https://pypi.python.org/pypi/pyflakes>`_ enabled (Preferences-> Editor-> Code Introspection/Analysis-> Code analysis (pyflakes) checked).
.. advice for emacs users ?

- Modules shouldn't become `God objects <http://en.wikipedia.org/wiki/God_object>`_. Keep them short (at most a few thousand lines) and well-organized.

- Commit changes before you go to sleep.
    You can always `rebase <https://help.github.com/articles/using-git-rebase/>`_ later multiple times, until you are happy with the history.
    This ensures that history won't have been forgotten by the time you return to that workspace.

- Prefix commits to classify the changes.
  The `NumPy development workflow <http://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html>`_ contains a summary of common abbreviations.
  You may prefer to use "MAI:" instead of "MAINT:", and "REF:" for refactoring.


Further reading, of general interest:

- "`On commit messages
  <http://who-t.blogspot.com/2009/12/on-commit-messages.html>`_" by Peter
  Hutterer (28 Dec 2009).

- `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html>`_

- Chapters 1, 2, 4, 6, 8 of the `Linux kernel coding style guide <https://www.kernel.org/doc/Documentation/CodingStyle>`_

- `The Power of 10: Rules for Developing Safety-Critical Code <http://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code>`_

- Chapter 1: "Style", `The Practice of Programming <http://www.cs.princeton.edu/~bwk/tpop.webpage/>`_
