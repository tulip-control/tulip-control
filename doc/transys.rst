Enumerated property representations
===================================

The subpackage ``tulip.transys`` (abbreviated to ``transys`` below)
contains classes for representing discrete relations with
data structures that are enumerated in the computer's memory.
This contrasts with symbolic data structures, discussed in the literature.

The core class is ``transys.labeled_graphs.LabeledDiGraph``,
itself a subclass of ``networkx.MultiDiGraph``.
The design philosophy is to deviate little from ``networkx``,
so it helps to familiarize oneself directly with ``networkx``.

The modules ``transys.mathset`` and ``transys.products`` are auxiliary,
and the subpackage ``transys.export`` contains functionality for
saving graphs in various formats, mainly formatting for layout by GraphViz.

The entry point for a user comprises of the ``transys`` modules:

- ``transys``
- ``automata``
- ``machines``

The mathematical objects representable with classes from these modules
can all be described uniformly and elegantly in the same mathematical language.
The only reason that different classes are available is for reasons of
implementation convenience for the user.

The following discussion is organized into parts.
First, we describe the class ``LabeledDiGraph``, because the above
modules contain its children.
Second, we describe the children in each module.


Labeled graphs
--------------

Already, ``networkx`` graphs `provide labeling capabilities
<http://networkx.github.io/documentation/latest/tutorial/tutorial.html#node-attributes>`_
for nodes and edges.
A ``dict`` is associated with each node and (multi)edge.
The labeling annotation can be stored as a key-value pair in the ``dict``.
What ``networkx`` does not provide is a mechanism for checking label consistency.

This is the main purpose of ``LabeledDiGraph``, via ``mathset.TypedDict``.
In the following, we call this consistency check "label type checking".

A ``LabeledDiGraph`` is associated with label types,
defined as arguments to the constructor.
Each label type is identified by a key, and is associated to values
that can be paired to the key.

Each graph node or edge can be annotated with a key-value pair.
If the key belongs to the label types, then the value must be
consistent with the type definition for that key.
For example:


.. code-block:: python

  from tulip.transys import labeled_graphs
  
  t = dict(name='color', values={'red', 'green', 'blue'},
           setter=True, default='green')
  g = labeled_graphs.LabeledDiGraph(node_label_types=[t])
  g.add_node(0, color='red')


Upon creation, each node is labeled with the ``default`` value,
here ``green``.

If we try to assign an invalid value to the typed key ``color``,
then an ``AttributeError`` will be raised.
In this case, the only typed key is ``'color'``.
You cannot use another key, unless you pass ``check=False``, as in


.. code-block:: python

  g.add_node(0, mode='on', check=False)


Using untyped keys like ``mode`` allows any key, as is normal in ``networkx``.
But arbitrarily named keys won't be recognized by ``transys``,
and the right keys with the wrong values will cause errors in ``tulip``.
This is the motivation for implementing typing.
Untyped keys are allowed for any additional annotation that
one may need to solve a particular problem.

Arguably, a most handy method is ``save``:


.. code-block:: python

  g.save('awesome.pdf')


``LabeledDiGraph`` is not intended to be instantiated itself,
but subclassed. The following sections consider the subclasses
present in the three main modules of ``transys``.
To define your own subclasses of ``LabeledDiGraphs``, read its docstring.
In that case, the constructors (``__init__``) of existing
subclasses can serve as examples.


Transition systems
------------------

A ``transys.transys.FiniteTransitionSystem`` describes a
set of sequences of nodes (states), as all the paths through it.
This set corresponds to a set of sequences of labels,
via the node and edge label annotations.
Usually, it is the latter set of sequences that is specified by
a temporal logic formula.

Each node is labeled with a set of atomic propositions,
owned by the player that governs the change of current node.
Each edge is labeled with:

- a system action (key ``'sys_actions'``)
- an environment action (key ``'env_actions'``)

If there is no environment, then the transition system describes
a "closed system" (only existentially quantified), with only
system actions. Otherwise, the transition system describes the
interaction between two players, an "open system", or game.

Viewing it as a game is an informal way of referring to
the existential and universal quantifiers that are later applied to
system and environment variables, respectively.


.. code-block:: python

  from tulip.transys import transys
  
  g = transys.FTS()
  g.atomic_propositions |= {'a', 'b'}
  g.add_node('s0', ap={'a'})
  #
  # 2 ways to change a label
  g.add_node('s0', ap={'b'})
  #
  # or
  g.node['s0']['ap'] = {'b'}


The method ``add_node`` overwrites the existing label,
so the label value ``{'a'}`` is replaced by ``{'b'}``.
The attribute ``atomic_propositions`` allows adding
more symbols to an existing set.

The argument-value pair ``ap={'a'}`` is used as a key-value
pair in the ``dict`` that stores the node's annotation.
An existing ``dict`` can also be passed, by unpacking, or
using the argument ``attr_dict``.

The annotation can be retrieved with:


.. code-block:: python

  annot = g.node['s0']['ap']


This assigns to ``annot`` the exact ``set`` object that labels
the node ``'s0'``. If no modification is intended, it is safer
to copy that set


.. code-block:: python

  r = g.node['s0']['ap']
  annot = set(r)


Attention is required, to avoid invalidating labels by mutation.
The label values are checked only through ``add_node`` or
setting of a value for ``TypedDict``. If we directly modify an
existing label value ``g.node['s0']['ap'].add('c')``,
then we can alter it to become invalid
(``'c'`` is not in ``atomic_propositions``).

To guard against such invalid values,
call the method ``LabeledDiGraph.is_consistent``,
which will detect any inconsistencies.
In the future, the integrated type checking may be
replaced completely with the flat approach of calling ``is_consistent``.

To avoid this issue altogether, labels can be modified as follows


.. code-block:: python
  
  from tulip.transys import transys

  g = transys.FTS()
  g.atomic_propositions |= {'a', 'b', 'c'}
  g.add_node('s0', ap={'a'})
  #
  # this does trigger type checking
  g.node['s0']['ap'] = g.node['s0']['ap'].union({'b', 'c'})
  #
  # equivalently
  r = g.node['s0']['ap']
  r = r.union({'b', 'c'})
  g.add_node('s0', ap=r)


The same mechanisms work for edges, but it is advisable to use
``LabeledDiGraph.transitions.find`` instead.
This avoids having to reason about the integer keys used internally by
``networkx`` to distinguish between edges with
the same pair of endpoint nodes (multi-edges).
A method ``LabeledDiGraph.states.find`` is available too.

The method ``LabeledDiGraph.transitions.find`` is intended as a tool
to slice the transition relation:

- find all edges from a given state
- find all edges to a given state
- find all edges with given label
- any combination of the above

For example, to find from state ``'s0'`` with ``sys_action = 'jump'`` all
possible post states,

.. code-block:: python

  set([e[1] for e in g.transitions.find('s0', with_attr_dict={'sys_action':'jump'})])

Alternatively ``find()`` may be bypassed in favor of the ``networkx`` method `edges_iter <https://networkx.github.io/documentation/latest/reference/generated/networkx.MultiDiGraph.edges_iter.html?highlight=edges_iter#networkx.MultiDiGraph.edges_iter>`_, as in

.. code-block:: python

  [u for u, d in g.edges_iter('s0', data=True) if d['sys_action'] == 'jump']

To add or label multiple nodes with one call,
call ``LabeledDiGraph.add_nodes_from``, as described `here
<http://networkx.github.io/documentation/latest/reference/generated/networkx.MultiDiGraph.add_nodes_from.html>`_.


.. code-block:: python

  nodes = range(3)
  #
  # multiple nodes, common label
  label = {'snow', 'north'}
  g.add_nodes_from([(u, dict(ap=label)) for u in nodes])
  #
  # multiple nodes, different labels
  labels = [{'a'}, {'a', 'b'}, {'b'}]
  g.add_nodes_from([(u, dict(ap=label)) for u, label in zip(nodes, labels)])


This might look cumbersome, but it becomes convenient for setting multiple labels:


.. code-block:: python

  g.add_edges_from(0, 1, env_actions='block', sys_actions='jump')



Automata
--------

There is no real difference between a "transition system" and an "automaton".
Both are ways of describing a set of sequences.
The only difference is that some parts of an automaton are omitted when
talking about a transition system, because they are trivial.

Currently, the automata in `transys` are "existential".
This means that a sequence belongs to the set described by an automaton,
if, and only if, there exists at least one satisfactory path through the
graph that represents the automaton.

What makes a path "satisfactory" doesn't have a fixed meaning: it depends.
To distinguish satisfactory from other paths, an *acceptance condition* is
made part of an automaton description.
It is common to call "accepting" a path that satisfies a given condition.

Traditionally, types of acceptance conditions have been associated with
names of people: Buchi, Rabin, Streett, Muller ("parity" is an exception).
Rabin and Streett correspond to the disjunctive and conjunctive
normal forms of a temporal logic formula.

In the ``tulip.transys.automata`` module you will find subclasses of
``LabeledDiGraph`` that are geared towards describing sets in the style
just mentioned. For example, the following code creates a Buchi automaton:


.. code-block:: python

  from tulip.transys import automata

  g = automata.BA()
  g.atomic_propositions.add_from(['a', 'b', 'c'])
  g.add_nodes_from(xrange(3))
  g.states.initial.add(2)
  g.states.accepting.add_from([1, 2])
  g.add_edge(2, 2, letter={'a'})
  g.add_edge(2, 0, letter={'b'})
  g.add_edge(0, 1, letter={'a'})
  g.add_edge(1, 1, letter={'a'})


A difference between transition systems and automata is
that the former usually have labeling on nodes, the latter on edges.
Historically, this is due to how algorithms for
enumerated model checking evolved. It is only a matter of
representation, not a feature of the sets that
these data structures describe.


Transducers (Machines)
----------------------

A transducer represents a function from finite sequences of input
(say symbols typed on a keyboard), to the next output (say screen color).
So, a transducer is an *implementation*, described in a way that is
executable (step-by-step). It differs from the above mainly programmatically.

If the design intent is described with a specification that is not
itself the implementation, then (automated) synthesis can construct
an implementation. Some forms of synthesis are available via ``tulip.synth``.
By convention, the constructed implementation is represented by ``machines.MealyMachine``.

The Mealy machine for producing a sequence of alternating 0s and 1s has the form:


.. code-block:: python

  from tulip.transys import machines

  g = machines.MealyMachine()
  g.add_inputs(dict(increment_index={1}))
  g.add_outputs(dict(sequence_element={0, 1}))
  g.add_nodes_from([0, 1])
  g.states.initial.add(0)
  g.add_edge(0, 1, increment_index=1, sequence_element=0)
  g.add_edge(1, 0, increment_index=1, sequence_element=1)


This Mealy machine is supposed to be "executed" as follows.
It starts at node ``0``. It reads an element from the input sequence.
The element is an assignment of values to identifiers.
Here, the only input identifier ("port") is ``'increment_index'``.
Then, the machine picks an edge labeled with the given input assignment.
The only such edge is ``(0, 1)``. The next node is ``1``.

The machine produces the next element of the output sequence.
This element is an assignment to output identifiers.
In our example, the assignment of value ``0`` to identifier ``'sequence_element'``.
Since we just started running the machine ``g``, this output assignment is
the first element in the output sequence.

You can get all this done with:


.. code-block:: python

  v, d = g.reaction(0, dict(increment_index=1))


which returns ``v = 1`` and ``d = dict(sequence_element=0)``.


The class ``machines.MooreMachine`` differs from a Mealy machine,
in that it imposes that element ``k`` of the output sequence
can depend only on elements *before* element ``k`` of the input sequence.
Use ``machines.MealyMachine``, because it is less restrictive.
