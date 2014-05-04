Benchmarks
==========

(Under construction!)  **This page concerns a previous release of TuLiP: version
0.4a or earlier.**

This page aims at summarizing some benchmark results to assess TuLiP's and
underlying algorithms' (i.e., solvers) capabilities to solve large scale
problems. We provide computation times and memory requirements as a function of
various (sometimes domain-specific) complexity measures.

Current TuLiP compatible solvers are:

* GR(1) game solvers: JTLV (default), gr1c
* Model Checkers: NuSMV, SPIN
* SMT solver: yices

Robot Motion Planning
---------------------

Gridworld Benchmarks
````````````````````

The first example here compares two model checkers, NuSMV and SPIN, used for
logic synthesis, in a robot motion planning example. We consider a robot that
moves in an nxm grid (complexity measure is grid size, i.e. n times m) with
randomly generated static obstacles (20% of the grid is filled with obstacles)
and tries to reach a goal position. The model checker either finds a path such
that the robot reaches the goal while avoiding the obstacles or declares that no
such path exists.

.. image:: spin-nusmv.png

Please see [S12]_ in the :doc:`bibliography` for additional comparisons between
NuSMV and SPIN with different sets of performance criteria and complexity
measures.

For the same robot motion planning example, the following graph shows that when
reactiveness to the environment is not required (e.g., when the environment is
static), using a model-checker for logic synthesis scales much better than a
more general game solver such as JTLV or gr1c.

.. image:: logic-synt-vs-game.png


Aircraft Electric Power Systems
-------------------------------

Untimed (static) Specifications
```````````````````````````````

Consider an electric power system base topology that includes both AC and DC
components. Each vertical set of components (AC generator, AC bus, rectifier
unit, DC bus, and two contactors) form a base unit. Figure 2 shows two base
units connected to each other, while additional units may be added, as
represented by the dotted lines. Table 1 compares the time it takes to convert a
set of primitives for a given base topology into formal specifications
compatible with Yices (Y) and TuLiP (T). Table 2 compares the time it takes for
Yices and TuLiP to solve/synthesize a controller for a given topology, as well
as the amount of memory.

.. csv-table:: Table 1: Specification Conversion Time for Yices (Y) and TuLiP (T) [time in seconds]
   :header: "Base Units", "Nodes", "Edges", "Conversion Time (Y/T)"

   4, 16, 18, .13/.11
   5, 20, 23, .25/.26
   10, 40, 48, 24/18
   12, 48, 58, 141/111
   15, 60, 73, 1634/1205

.. csv-table:: Table 2: Comparison of Synthesis Time for Yices (Y) and TuLiP (T). [time in seconds]
   :header: "Base Units", "Yices Env.", "Time (Y/T)", "Mem. (Y/T)"

   4, 25, .25/10.7, 25MB/215MB
   5, 36, .82/1015, 36MB/16GB
   10, 121, 205.7/-, 53MB/-
   12, 169, 1410/-, 158MB/-
   15, 256, 62208/-, 1.2GB/-

.. figure:: topology.png
   :alt: Part of an AES topology showing two vertical sets of components, each with AC generator, AC bus, rectifier unit, DC bus, and two contactors.

   Figure 2

Timed Specifications
````````````````````

For the topology in Figure 3, Table 3 lists the automaton size as well as total
synthesis time while varying the number of clocks, as well as the discretization
of clock "ticks." The first column indicates the number of clocks, or counters,
used in the synthesis problem. One clock refers to one essential bus that can
never be unpowered for more than some number of ticks. The second column thus
indicates the maximum time in which the essential bus can be unpowered (i.e.,
how large the discretized clock space must be). The higher the number, the more
clock ticks must be incorporated into the synthesis problem. The third and
fourth columns refer to the total automaton size (i.e., number of states)
generated, as well as the total computation time (in seconds), respectively.

.. csv-table:: Table 3
   :header: "No. of Clocks", "Clock \"Ticks\"", "Aut. Size", "Time [sec]"
   
   1, 1, 32, 1.5
   1, 3, 64, 1.7
   1, 5, 96, 1.7
   1, 10, 176, 2.8
   1, 20, 336, 3.1
   2, 1, 79, 2
   2, 3, 96, 2
   2, 5, 224, 2.1
   2, 10, 384, 2.5
   2, 20, 704, 2.5
   3, 1, 478, 3.5
   3, 3, 2858, 7
   3, 5, 7180, 160
   3, 10, 45492, 1084
   3, 20, 88604, 4796
   4, 1, 1798, 7.2
   4, 3, 22008, 308
   4, 5, 93386, 4778

.. figure:: topology2.png
   :alt: AES topology with four generator-contactor-bus columns, where the buses are connected in series by contactors.

   Figure 3
