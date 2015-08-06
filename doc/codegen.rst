Code generation and exporting of controllers
============================================

Besides the various routines for analysis and execution of controllers
constructed with TuLiP, it is also possible to automatically generate
implementations in formats appropriate for use in other contexts.  We refer to
this process as *exporting*.  Here we provide descriptions and examples of the
various exporting routines available.

``python_case``
---------------

Given a Mealy machine (Sec. 3.3 in `[LS11] <bibliography.html#ls11>`_) as would be obtained, for example, from an invocation of
``tulip.synth.synthesize``, the function ``python_case`` in ``tulip.dumpsmach``
generates an implementation as a standalone Python class.  The class implements
the machine by

* tracking the current state of the machine and
* providing a ``move`` method that accepts the input of the machine and returns
  the appropriate output as a dictionary in which the output variable names are
  keys.

Calls to ``move`` cause the internal state to transition.  The machine can be
reset to its initial state by manually calling the ``__init__`` method or
effectively by creating a new instance. Note that the internal state is entirely
represented by the attribute ``state``. Thus it is possible to save a copy of
the current state of the machine and return to it later. E.g., the output that
would be obtained if some inputs were applied can be discovered by the following
idiom.

.. code-block:: python

  import copy
  saved_state = copy.copy(M.state)
  sample_outputs = M.move(**sample_inputs)
  M.state = saved_state

The generated code does not depend on ``tulip``; that is, it can run without
TuLiP being installed.  As such, we refer to it as being "standalone".

Example
```````

Consider the script ``examples/robot_planning/gr1.py`` as distributed with
TuLiP.  At the end of the script, there is an object named ``ctrl`` that is an
instance of ``MealyMachine``.  It represents the controller that was constructed
automatically (or "synthesized") according to the specification.  Append the
following code to the end of the script ``gr1.py``.  (Alternatively, if IPython
is installed, then first run ``ipython -i gr1.py``, and then enter the
following.)

.. code-block:: python

  from tulip import dumpsmach
  dumpsmach.write_python_case("gr1controller.py", ctrl, classname="ExampleCtrl")

The first line merely loads the ``dumpsmach`` module.  The second line calls a
function that, in turn, calls ``python_case`` and saves the result to a file
named "gr1controller.py".  That file contains a class named "ExampleCtrl" that
implements ``ctrl``, as demonstrated by the following script.

.. code-block:: python

  from __future__ import print_function
  from gr1controller import ExampleCtrl

  M = ExampleCtrl()
  print('In order, the input variables: '+', '.join(M.input_vars))
  for i in xrange(10):
      input_values = {"park": 0}
      print(M.move(**input_values))

Alternatively, the above code can be modified by calling ``M.move`` with the
input variable names as keyword parameters::

  print(M.move(park=0))
