Developer's Guide
=================

The purpose of this page is to provide guidelines for contributors to the TuLiP
project.  Also consult the `Developers' Wiki <https://github.com/tulip-control/tulip-control/wiki>`_ and the `tulip-control-discuss mailing list <https://sourceforge.net/p/tulip-control/mailman/tulip-control-discuss/>`_ (members only).

Organization and Rules
----------------------

We begin with important organizational notes and **rules** that should
be followed:

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

- When referring to publications, check for a corresponding entry in
  ``doc/bibliography.rst`` and create one if needed, following the `BibTeX
  alpha.bst style <http://sites.stat.psu.edu/~surajit/present/bib.htm#alpha>`_.
  References in the Sphinx-built documentation are as usual, e.g.,
  ``[WTOXM11]_``.  References in docstrings (in the code) should be to the URL
  of the corresponding entry on the TuLiP website, using `Epydoc syntax
  <http://epydoc.sourceforge.net/manual-epytext.html>`_, e.g., ::

    U{[WTOXM11] <http://tulip-control.sourceforge.net/doc/bibliography.html#wtoxm11>}

- A script for running tests is ``run_test.py`` in the root of the source tree.
  It searches under the directory ``tests/`` for files with names ending in
  "_test.py", and passes these to `nose <http://readthedocs.org/docs/nose/>`_.
  Use the flag "-h" to see driver script options.  Extra details about options:

  * The flag "--cover" to generate a coverage report, which will likely be
    placed under ``tests/cover/``.  It uses `Ned Batchelder's coverage module
    <http://www.nedbatchelder.com/code/modules/coverage.html>`_.

  * The flag "--outofsource" will cause ``tulip`` to be imported from outside
    the current directory.  This is useful for testing against the installed
    form of TuLiP.

Version naming
--------------

(Copied verbatim from an email sent by Richard Murray on the TuLiP-discuss mailing list on 17 May 2011.)

For version numbers, the style I like is N.mx where

* N = major version; everything that worked before could break
* m = revision; most functions should work, but might need (minor) modifications
* x = minor revision; code that ran should continue running

So, if you go from version 1.2c to 1.2d, then no interfaces should
change, code should continue to run, etc.  If you go from 1.2d to
1.3a, then there might be changes in some arguments lists or other
small things, but previous functionality still in place (somewhow).
If you go from 1.3a to 2.0a, then we can make whatever changes we
want.

None of these version numbers would go in individual files, but would
be a label for the entire package.


Advice
------

The following are software engineering best practices that you should try to
follow.  We mention them here for convenience of reference and to aid new
committers.

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

Further reading, of general interest:

- "`On commit messages
  <http://who-t.blogspot.com/2009/12/on-commit-messages.html>`_" by Peter
  Hutterer (28 Dec 2009).

- Chapters 1, 2, 4, 6, 8 of the `Linux kernel coding style guide <https://www.kernel.org/doc/Documentation/CodingStyle>`_

- `The Power of 10: Rules for Developing Safety-Critical Code <http://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code>`_

- Chapter 1: "Style", `The Practice of Programming <http://cm.bell-labs.com/cm/cs/tpop/>`_
