Enumerated property representations
===================================

The subpackage `tulip.transys` (abbreviated to `transys` below)
contains classes for representing discrete relations with
data structures that are enumerated in the computer's memory.
This contrasts with symbolic data structures, discussed in the literature.

The core class is `transys.labeled_graphs.LabeledDiGraph`,
itself a subclass of `networkx.MultiDiGraph`.
The design philosophy is to deviate little from `networkx`,
so it helps to familiarize oneself directly with `networkx`.

The modules `transys.mathset` and `transys.products` are auxiliary,
and the subpackage `transys.export` contains functionality for
saving graphs in various formats, mainly formatting for layout by GraphViz.

The entry point for a user comprises of the `transys` modules:

  - `transys`
  - `automata`
  - `machines`

The mathematical objects representable with classes from these modules
can all be described uniformly and elegantly in the same mathematical language.
The only reason that different classes are available is for reasons of
implementation convenience for the user.

The following discussion is organized into parts.
First, we describe the class `LabeledDiGraph`, because the above
modules contain its children.
Second, we describe the children in each module.


Labeled graphs
--------------

Already, `networkx` graphs `provide labeling capabilities
<http://networkx.github.io/documentation/latest/tutorial/tutorial.html#node-attributes>`_
for nodes and edges.
A `dict` is associated with each node and (multi)edge.
The labeling annotation can be stored as a key-value pair in the `dict`.
What `networkx` does not provide is a mechanism for checking label consistency.

This is the main purpose of `LabeledDiGraph`, via `mathset.TypedDict`.
In the following, we call this consistency check "label type checking".

A `LabeledDiGraph` is associated with label types,
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


Upon creation, each node is labeled with the `default` value,
here `green`.

If we try to assign an invalid value to the typed key `color`,
then an `AttributeError` will be raised.
In this case, the only typed key is `'color'`.
You cannot use another key, unless you pass `check=False`, as in


.. code-block:: python

  g.add_node(0, mode='on', check=False)


Using untyped keys like `mode` allows any key, as is normal in `networkx`.
But arbitrarily named keys won't be recognized by `transys`,
and the right keys with the wrong values will cause errors in `tulip`.
This is the motivation for implementing typing.
Untyped keys are allowed for any additional annotation that
one may need to solve a particular problem.

`LabeledDiGraph` is not intended to be instantiated itself,
but subclassed. The following sections consider the subclasses
present in the three main modules of `transys`.
To define your own subclasses of `LabeledDiGraphs`, read its docstring.
In that case, the constructors (`__init__`) of existing
subclasses can serve as examples.


Transition systems
------------------

A `transys.transys.FiniteTransitionSystem` describes a
set of sequences of nodes (states), as all the paths through it.
This set corresponds to a set of sequences of labels,
via the node and edge label annotations.
Usually, it is the latter set of sequences that is specified by
a temporal logic formula.

Each node is labeled with a set of atomic propositions,
owned by the player that governs the change of current node.
Each edge is labeled with:

- a system action (key `'sys_actions'`)
- an environment action (key `'env_actions'`)

If there is no environment, then the transition system describes
a "closed system" (only existentially quantified), with only
system actions. Otherwise, 


.. code-block:: python

  from tulip.transys import transys
  
  g = transys.FTS()
  g.atomic_propositions |= {'a', 'b'}
  g.add_node('s0', ap={'a'})
  
  # 2 ways to change a label
  g.add_node('s0', ap={'b'})
  
  # or
  g.node['s0']['ap'] = {'b'}


The method `add_node` overwrites the existing label,
so the label value `{'a'}` is replaced by `{'b'}`.
The attribute `atomic_propositions` allows adding
more symbols to an existing set.

The argument-value pair `ap={'a'}` is used as a key-value
pair in the `dict` that stores the node's annotation.
An existing `dict` can also be passed, by unpacking, or
using the argument `attr_dict`.

The annotation can be retrieved with:


.. code-block:: python

  annot = g.node['s0']['ap']


This assigns to `annot` the exact `set` object that labels
the node `'s0'`. If no modification is intended, it is safer
to copy that set


.. code-block:: python

  r = g.node['s0']['ap']
  annot = set(r)


Attention is required, to avoid invalidating labels by mutation.
The label values are checked only through `add_node` or
setting of a value for `TypedDict`. If we directly modify an
existing label value `g.node['s0']['ap'].add('c')`,
then we can alter it to become invalid
(`'c'` is not in `atomic_propositions`).

To guard against such invalid values,
call the method `LabeledDiGraph.is_consistent`,
which will detect any inconsistencies.
In the future, the integrated type checking may be
replaced completely with the flat approach of calling `is_consistent`.

To avoid this issue altogether, labels can be modified as follows


.. code-block:: python

  # this does trigger type checking
  g.node['s0']['ap'] = g.node['s0']['ap'].union({'b', 'c'})
  # equivalently
  r = g.node['s0']['ap']
  r = r.union({'b', 'c'})
  g.add_node('s0', ap=r)


The same mechanisms work for edges, but it is advisable to use
`LabeledDiGraph.transitions.find` instead.
This avoids having to reason about the integer keys used internally by
`networkx` to distinguish between edges with
the same pair of endpoint nodes (multi-edges).
A method `LabeledDiGraph.states.find` is available too.

The method `LabeledDiGraph.transitions.find` is intended as a tool
to slice the transition relation:

  - find all edges from a given state
  - find all edges to a given state
  - find all edges with given label
  - any combination of the above


To add or label multiple nodes with one call,
call `LabeledDiGraph.add_nodes_from`, as described `here
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



Transducers (Machines)
----------------------
