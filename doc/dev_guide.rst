Developer's Guide
=================

The purpose of this page is to provide guidelines for contributors to the TuLiP
project.  We begin with important organizational notes and **rules** that should
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
	Break them down into simpler ones. When possible chain them, so that they are checked in sequence.
	This way a function returns when a condition is False, so the conjunction is implicit and easier to follow, one check at a time.

- Name things to minimize comments.
	Comments are useless if they attempt to explain what the code evidently does and can be harmful if they fail to do so and instead describe what it was intended to do, giving a false impression of correctness.

- Have (simple) static checking on.
	e.g. `Spyder <http://code.google.com/p/spyderlib/>`_ with `pyflakes <https://pypi.python.org/pypi/pyflakes>`_ enabled (Preferences-> Editor-> Code Introspection/Analysis-> Code analysis (pyflakes) checked).
.. advice for emacs users ?

- Modules shouldn't become `God objects <http://en.wikipedia.org/wiki/God_object>`_. Keep them short (at most a few thousand lines) and well-organized.  Avoid `this <https://github.com/mdipierro/gluino/blob/master/gluino/dal.py>`_.

Further reading, of general interest:

- Chapters 1, 2, 4, 6, 8 of the `Linux kernel coding style guide <https://www.kernel.org/doc/Documentation/CodingStyle>`_

- `The Power of 10: Rules for Developing Safety-Critical Code <http://en.wikipedia.org/wiki/The_Power_of_10:_Rules_for_Developing_Safety-Critical_Code>`_

- Chapter 1: "Style", `The Practice of Programming <http://cm.bell-labs.com/cm/cs/tpop/>`_
