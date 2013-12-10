Tutorial and Examples
=======================

TuLiP is developed for synthesis of discrete controllers for hybrid control
systems, including supervisory controllers, switching controllers and
receding horizon temporal logic planners.

Discrete State Example
----------------------
To illustrate the basic synthesis capabilities of TuLiP, we synthesize a
simple discrete state controller for a robotic motion control system.

Synthesis of Reactive Controllers
---------------------------------
We consider a system that comprises the physical component, which we refer
to as the plant, and the (potentially dynamic and not a priori known)
environment in which the plant operates.  The system may contain both
continuous (physical) and discrete (computational) components.  In summary,
the problem we are interested in consists of

  - discrete system state,
  - continuous system state,
  - (discrete) environment state, and
  - specification.

Here, `discrete` state refer to the state that can take only a finite number
of possible values while `continuous` state refer to the state that can take
an infinite number of possible values, e.g., the position of the car.  The
`environment` state is related to factors over which the system does not
have control such as the position of an obstacle and the outside
temperature.  At any given time, the controller regulates the `system` (or
`controlled`) state such that the specification is satisfied, given the
current value of the environment variables and the previous system states.
We say that the specification is `realizable` if for any possible behavior
of the environment, such a controller exists, i.e., there exists a strategy
for the system to satisfy the specification.

Suppose the continuous state of the system evolves according to the
following discrete-time linear time-invariant state space model:
for :math:`t \in \{0,1,2,...\}`

.. math::
   s[t+1]  =   As[t] + Bu[t] + Ed[t] + K \qquad
   u[t] \in U,\, d[t] \in D,\, s[0] \in S,
   :label: dynamics

where :math:`S \subseteq \mathbb{R}^n` is the state space of the continuous 
component of the system, 
:math:`U \subseteq \mathbb{R}^m` is the set of admissible control inputs, 
:math:`D \subseteq \mathbb{R}^p` is the set of exogenous disturbances and
:math:`s[t], u[t], d[t]` are the continuous state, the control signal and
the exogenous disturbance, respectively, at time :math:`t`.

We consider the case where the sets :math:`S, U, D` are bounded polytopes.

Let :math:`\Pi` be a finite set of atomic propositions of system variables.
Each of the atomic propositions in :math:`\Pi` essentially captures the
states of interest.
We consider the specification of the form

.. math::
   \varphi = \big(\varphi_{init} \wedge \varphi_e) \implies \varphi_s.
   :label: spec

Here, the assumption :math:`\varphi_{init}` on the initial condition of the system
is a propositional formula built from :math:`\Pi.`
The assumption :math:`\varphi_e` on the environment and the desired behavior 
:math:`\varphi_s` are LTL formulas built from :math:`\Pi.`

As described in the :doc:`intro`, our approach to this reactive control 
system synthesis consists of the following main steps:

   1. :ref:`Generate a proposition preserving partition of the continuous
      state space. <ssec:prop-part>` 
   2. :ref:`Discretize the continuous state space based on the evolution of
      the continuous state. <ssec:disc>` 
   3. :ref:`Digital design synthesis. <ssec:syn>`

.. _ssec:prop-part:

Proposition Preserving Partition of Continuous State Space
``````````````````````````````````````````````````````````
Given the continuous state space :math:`S` of the system and the set
:math:`\Pi_c` of propositions on the continuous state of the system, we
partition :math:`S` into a finite number of cells such that all the
continuous states in each cell satisfy exactly the same set of propositions
in :math:`\Pi_c`.

This can be done using the following function call:

     .. autofunction:: abstract.prop2part2
	:noindex:

The above function returns a proposition preserving partition as a PropPreservingPartition object.

     .. autoclass:: abstract.PropPreservingPartition
	:noindex:


.. _ssec:disc:

Continuous State Space Discretization
`````````````````````````````````````
Given a proposition preserving partition of the continuous state space and
the evolution of the continuous state as in :eq:`dynamics`,
we refine the partition based on the reachability relation between cells
and obtain a finite state abstraction of the evolution of the continuous state, 
represented by a finite transition system.

CtsSysDyn class is used to define continuous dynamics.

     .. autoclass:: discretize.CtsSysDyn
	:noindex:


Once we have the proposition preserving partition and the continuous dynamics,
continuous state space discretization can be done using the following function call:

     .. autofunction:: discretize.discretizeM
	:noindex:


.. _ssec:syn:

Digital design synthesis
````````````````````````

The continuous state space discretization generates a finite state abstraction
of the continuous state, represented by a finite transition system.
Each state in this finite transition system corresponds to a cell in the continuous
domain.
A transition :math:`c_i \to c_j` in this finite state system indicates that 
from any continuous state :math:`s_0` that belongs to cell :math:`c_i`, 
there exists a sequence of control inputs :math:`u_0, u_1, \ldots, u_{N-1}` 
that takes the system to another continuous state :math:`s_{N}` in cell :math:`c_j`.
Hence, under the assumption that the specification is stutter invariant,
we can describe the continuous dynamics by an LTL formula of the form

.. math::
   (v = c_i) \implies next(\bigvee_{j \text{ s.t. } c_i \to c_j} v = c_j),

where :math:`v` is a new discrete variable that describes in which cell
the continuous state is.

Since the partition is proposition preserving, all the continuous states that belong
to the same cell satisfy exactly the same set of propositions on the continuous
state. By the abuse of notation, we write :math:`c_j \models X_i` if all the continuous
states in cell :math:`c_j` satisfy proposition :math:`X_i`.
Then, we can replace any proposition :math:`X_i` on the continuous state variables
by the formula :math:`\displaystyle{\bigvee_{j \text{ s.t. } c_j \models X_i} v = c_j}`.

Putting everything together, we now obtain a specification of the form
in :eq:`spec` (see also :ref:`ssec:spectips`).  We can then use the
GR(1) Game implementation in `JTLV <http://jtlv.ysaar.net/>`_ to
automatically synthesize a planner that ensures the satisfaction of
the specification, taking into account all the possible behaviors of
the environment.  This can be done using the following steps.

    1. Generate input to JTLV

        .. autofunction:: jtlvint.generateJTLVInput
	   :noindex:

    2. Synthesize the discrete planner

        .. autofunction:: jtlvint.computeStrategy
	   :noindex:

    3. Construct the automaton

        .. autoclass:: automaton.Automaton
	   :noindex:


Step 1 and 2 above can be combined using the following function:
        .. autofunction:: jtlvint.synthesize
	   :noindex:


.. _ssec:spectips:

Syntax for Writing Specifications 
`````````````````````````````````

The specification :eq:`spec` may contain various LTL operators, arithmetic operators for integer variables and parentheses 
for precedence or to increase readability. Here is a quick list of names of the operators followed by the corresponding symbols that 
are compatible with rhtlp: 

============================  =====================
     Operator                      Symbol
----------------------------  ---------------------
and                                   &
or                                    \|
not                                   ! 
next                                next
always                               []
eventually                           <>
implies                              -> 
boolean constants               TRUE, FALSE 
order for integer variables    =, <, >, >=, <= 
arithmetic operations                +, - 
parentheses                          (, )
============================  =====================

Also note that the domains of the variables can be either boolean or a list of integers.

.. _ssec:ex1:

Example 1: Robot Motion Planning with only Discrete Decisions
`````````````````````````````````````````````````````````````
This example is provided in examples/robot_discrete_simple.py.
It illustrates the use of the jtlvint module in synthesizing a planner
for a robot that only needs to make discrete decision.

.. image:: robot_discrete_simple.*
   :align: center

We consider the robot moving around the regions as shown in the above figure
while receiving externally triggered park signal.
The specification of the robot is

.. math::
   \varphi = \square \diamond(\neg park) \implies (\square \diamond(s \in C_5)
   \wedge \square(park \implies \diamond(s \in C_0))).

We cannot, however, deal with this specification directly since it is not in the form of 
GR[1].
An equivalent GR[1] specification of the above specification can be obtained
by introducing an auxiliary discrete system variable :math:`X0reach,` initialized to 
`True`. The transition relation of :math:`X0reach,` is given by
:math:`\square(\text{next}(X0reach) = (s \in C_0 \vee (X0reach \wedge \neg park))).`

To automatically synthesize a planner for this robot, we first import the necessary modules.

.. highlight:: python

.. literalinclude:: ../examples/robot_discrete_simple.py
   :start-after: @import_section@
   :end-before: @import_section_end@

Specify the smv file, spc file and aut file.

.. literalinclude:: ../examples/robot_discrete_simple.py
   :start-after: @filename_section@
   :end-before: @filename_section_end@

Specify the environment variables.

.. literalinclude:: ../examples/robot_discrete_simple.py
   :start-after: @envvar_section@
   :end-before: @envvar_section_end@

Specify the discrete system variable.

.. literalinclude:: ../examples/robot_discrete_simple.py
   :start-after: @sysdiscvar_section@
   :end-before: @sysdiscvar_section_end@

Specify the transition system representing the continuous dynamics.
First, we list the propositions on the continuous states.
Here, these propositions specify in which cell the robot is, 
i.e., Xi means that the robot is in cell Ci.
Then, we specify the regions.
Note that the first argument of Region(poly, prop) should be a list of 
polytopes. But since we are not dealing with the actual controller, we will 
just fill it with a string (think of it as a name of the region).
The second argument of Region(poly, prop) is a list that specifies which 
propositions in cont_props above is satisfied. As specified below, regioni 
satisfies proposition Xi.
Finally, we specify the adjacency between regions. 
disc_dynamics.adj[i][j] = 1 if starting from region j,
the robot can move to region i while only staying in the union of region i 
and region j.

.. literalinclude:: ../examples/robot_discrete_simple.py
   :start-after: @ts_section@
   :end-before: @ts_section_end@

Specification.

.. literalinclude:: ../examples/robot_discrete_simple.py
   :start-after: @specification@
   :end-before: @specification_end@

Generate input to JTLV.

.. literalinclude:: ../examples/robot_discrete_simple.py
   :start-after: @geninput@
   :end-before: @geninput_end@

Check realizability.

.. literalinclude:: ../examples/robot_discrete_simple.py
   :start-after: @check@
   :end-before: @check_end@

Construct an automaton.

.. literalinclude:: ../examples/robot_discrete_simple.py
   :start-after: @compaut@
   :end-before: @compaut_end@

Run simulation.

.. literalinclude:: ../examples/robot_discrete_simple.py
   :start-after: @sim@
   :end-before: @sim_end@


.. _ssec:ex2:

Example 2: Robot Motion Planning
````````````````````````````````
This example is provided in examples/robot_simple.py.
It is an extension of the previous example by including continuous dynamics.

First, we import the necessary modules, 
specify the smv file, spc file and aut file,
and specify the environment and the discrete system variables
as in the previous example.

.. literalinclude:: ../examples/robot_simple.py
   :start-after: @importvar@
   :end-before: @importvar_end@

Next, we specify the continuous dynamics.
This includes specifying the continuous state space, propositions on continuous variables,
and the dynamics.
The robot dynamics in this case is :math:`\dot{x} = u_x, \dot{y} = u_y.`

.. literalinclude:: ../examples/robot_simple.py
   :start-after: @contdyn@
   :end-before: @contdyn_end@

Now, we can construct the proposition preserving partition of the continuous state space
and discretize the continuous state space based on the dynamics.

.. literalinclude:: ../examples/robot_simple.py
   :start-after: @discretize@
   :end-before: @discretize_end@

The rest is the same as in the previous example. 
We specify system specification, 
generate input to JTLV,
check realizability,
construct an automaton, run simulation and write simulation results to a file.

.. literalinclude:: ../examples/robot_simple.py
   :start-after: @gencheckcomp@
   :end-before: @gencheckcomp_end@
.. literalinclude:: ../examples/robot_simple.py
   :start-after: @sim@
   :end-before: @sim_end@

Defining a Synthesis Problem
````````````````````````````

SynthesisProb class provides a self-contained structure for defining 
an embedded control software synthesis problem.
It contains several useful functions that allow the problem to be solved in one shot,
combining the 3 steps as previously described.

     .. autoclass:: rhtlp.SynthesisProb
	:noindex:


Example 3: Robot Motion Planning using SynthesisProb
````````````````````````````````````````````````````
This example is provided in examples/robot_simple2.py.
It is exactly the same problem as :ref:`Example 2 <ssec:ex2>`
but solved using SynthesisProb, instead of the jtlvint module.

First, we import the necessary modules
and specify the environment and the discrete system variables and
the continuous dynamics.
Note that we don't have to specify the smv file, spc file and aut file
as in the previous examples.

.. literalinclude:: ../examples/robot_simple2.py
   :start-after: @importvardyn@
   :end-before: @importvardyn_end@

Next, we specify system specification. Here, specification is a GRSpec object,
instead of a list of length 2 as in the previous examples.

.. literalinclude:: ../examples/robot_simple2.py
   :start-after: @specification@
   :end-before: @specification_end@

Now, we have all the necessary elements to construct a synthesis problem.

.. literalinclude:: ../examples/robot_simple2.py
   :start-after: @synprob@
   :end-before: @synprob_end@

Once a SynthesisProb object is constructed, we can check the realizability of this problem
and construct the automaton.

.. literalinclude:: ../examples/robot_simple2.py
   :start-after: @checkcomp@
   :end-before: @checkcomp_end@

Finally, we can run the simulation as before.

.. literalinclude:: ../examples/robot_simple2.py
   :start-after: @sim@
   :end-before: @sim_end@

Working with Systems with Piecewise Affine Dynamics
```````````````````````````````````````````````````

TuLiP can also handle piecewise affine dynamics of the form: 

for :math:`t \in \{0,1,2,...\}`

.. math::
   s[t+1]  &=   A_is[t] + B_iu[t] + E_id[t] + K_i\\
   u[t]    &\in U_i\\
   d[t]    &\in D_i\\
   s[t]	   &\in S_i
   :label: pwadynamics

where :math:`S_i \subseteq \mathbb{R}^n` for :math:`i \in \{0,1,2, \ldots,n_s\}` form 
a polytopic partition of the state space :math:`S`, in :eq:`dynamics`,
:math:`U_i \subseteq \mathbb{R}^m` is the set of admissible control inputs, 
:math:`D_i \subseteq \mathbb{R}^p` is the set of exogenous disturbances within :math:`S_i`, and
:math:`s[t], u[t], d[t]` are the continuous state, the control signal and
the exogenous disturbance, respectively, at time :math:`t`.

PwaSubsysDyn class is used to represent subsystems of the form :eq:`pwadynamics`.

     .. autoclass:: discretize.PwaSubsysDyn
	:noindex:

The subsystems can be put together to define a piecewise affine system which is represented 
by PwaSysDyn class.

     .. autoclass:: discretize.PwaSysDyn
	:noindex:


Example 4: Robot Motion Planning with Piecewise Affine Dynamics
```````````````````````````````````````````````````````````````

This example is provided in examples/robot_simple_pwa.py.
It is an extension of the previous examples including a robot model
with piecewise affine dynamics.

Assume our robot is traveling on a nonhomogenous surface (x-y plane), 
resulting in different dynamics at different parts of the plane. 
Since the continuous state space in this example is just x-y position, different
dynamics in different parts of the surface can be modeled as a piecewise 
affine system. When :math:`s[t] \in[0, 3]\times[0.5, 2]`, the following dynamics
are active:

.. literalinclude:: ../examples/robot_simple_pwa.py
   :start-after: @subsystem0@
   :end-before: @subsystem0_end@

When :math:`s[t] \in[0, 3]\times[0, 0.5]`, the following dynamics
are active:

.. literalinclude:: ../examples/robot_simple_pwa.py
   :start-after: @subsystem1@
   :end-before: @subsystem1_end@

Piecewise affine system can be formed from the dynamics of its subsystems.

.. literalinclude:: ../examples/robot_simple_pwa.py
   :start-after: @pwasystem@
   :end-before: @pwasystem_end@

Discretization and synthesis follow exactly as before.

.. literalinclude:: ../examples/robot_simple_pwa.py
   :start-after: @synth@
   :end-before: @synth_end@

Finally, we can simulate the continuous and discrete parts and plot the resulting trajectories.

.. literalinclude:: ../examples/robot_simple_pwa.py
   :start-after: @sim@
   :end-before: @sim_end@

Receding Horizon Temporal Logic Planning
----------------------------------------

For systems with a certain structure, the computational complexity of
the planner synthesis can be alleviated by solving the planning problems
in a receding horizon fashion, i.e., compute the plan or strategy over
a "shorter" horizon, starting from the current state,
implement the initial portion of the plan,
move the horizon one step ahead, and recompute.
This approach essentially reduces the planner synthesis problem
into a set of smaller problems.
To ensure that this "receding horizon" framework preserves
the desired system-level temporal properties, certain sufficient conditions
need to be satisfied.

We consider a specification of the form

.. math::
   \varphi = \big(\psi_{init} \wedge \square \psi_e^e \wedge 
   \bigwedge_{i \in I_f} \square\diamond \psi_{f,i}\big) \implies
   \big(\square \psi_s \wedge \bigwedge_{i \in I_g} \psi_{g,i}\big).
   :label: GR1Spec

Given a specification of this form, we first construct a finite state abstraction
of the physical system.
Then, for each :math:`i \in I_g`, we organize the system states
into a partially ordered set :math:`\mathcal{P}^i = (\{\mathcal{W}_j^i\}, \preceq_{\psi_{g,i}})`
where :math:`\mathcal{W}_0^i` are the set of states satisfying
:math:`\psi_{g,i}.`
For each :math:`j`, we define a short-horizon specification :math:`\Psi_j^i`
associated with :math:`\mathcal{W}_j^i` as

.. math::
   \Psi_j^i = \big((\nu \in \mathcal{W}_j^i) \wedge \Phi \wedge \square \psi_e^e \wedge 
   \bigwedge_{k \in I_f} \square\diamond \psi_{f,k}\big) \implies
   \big(\square \psi_s \wedge \square\diamond(\nu \in \mathcal{F}^i(\mathcal{W}_j^i)) \big).

Here, :math:`\Phi` describes receding horizon invariants.
:math:`\Phi` needs to be defined such that :math:`\psi_{init} \implies \Phi` is a tautology.
:math:`\mathcal{F}^i : \{\mathcal{W}_j^i\} \to \{\mathcal{W}_j^i\}` is a mapping
such that :math:`\mathcal{F}^i(\mathcal{W}_j^i) \prec_{\psi_{g,i}} \mathcal{W}_j^i, \forall j \not= 0.`
:math:`\mathcal{F}^i(\mathcal{W}_j^i)` essentially defines intermediate goal for starting in
:math:`\mathcal{W}^i_j.`

.. image:: rhtlp_strategy.*

The above figure provides a graphical description of the receding horizon strategy 
for a special case where for each 
:math:`i \in I_g, \mathcal{W}^i_j \prec_{\psi_{g,i}} \mathcal{W}^i_k, \forall j < k`,
:math:`\mathcal{F}^i(\mathcal{W}^i_j) = \mathcal{W}^i_{j-1}, \forall j > 0` 
and :math:`F^i(\mathcal{W}^i_0) = \mathcal{W}^i_0.`
Please refer to our `paper <http://www.cds.caltech.edu/~nok/doc/tac10.pdf>`_ for more details.


Short-Horizon Problem
`````````````````````

A short-horizon problem can be defined using the ShortHorizonProb class.

     .. autoclass:: rhtlp.ShortHorizonProb
	:noindex:


Receding Horizon Temporal Logic Planning Problem
````````````````````````````````````````````````

A receding horizon temporal logic planning problem contains a collection of
short-horizon problems. It can be defined using the RHTLPProb class, which
contains many functions such as ``computePhi()`` and ``validate()``.   

     .. autoclass:: rhtlp.RHTLPProb
	:members: computePhi, validate
	:noindex:



Example 5: Autonomous Vehicle
`````````````````````````````
This example is provided in examples/autonomous_car_road.py.
It is a simplified version of the problem presented in our
`CDC paper <http://www.cds.caltech.edu/~nok/doc/cdc09.pdf>`_

We first import the necessary modules.

.. literalinclude:: ../examples/autonomous_car_road.py
   :start-after: @import_section@
   :end-before: @import_section_end@

Specify the road configuration and the problem setup.

.. literalinclude:: ../examples/autonomous_car_road.py
   :start-after: @roadsetup@
   :end-before: @roadsetup_end@

Continuous dynamics.

.. literalinclude:: ../examples/autonomous_car_road.py
   :start-after: @contdyn@
   :end-before: @contdyn_end@

Variables and propositions.

.. literalinclude:: ../examples/autonomous_car_road.py
   :start-after: @varprop@
   :end-before: @varprop_end@

Specification.

.. literalinclude:: ../examples/autonomous_car_road.py
   :start-after: @spec@
   :end-before: @spec_end@

Now, we construct the RHTLPProb object.

.. literalinclude:: ../examples/autonomous_car_road.py
   :start-after: @prob@
   :end-before: @prob_end@

Add ShortHorizonProb objects to the RHTLPProb object.

.. literalinclude:: ../examples/autonomous_car_road.py
   :start-after: @shorthoriz@
   :end-before: @shorthoriz_end@

For each :math:`\mathcal{W}_j`, set :math:`\mathcal{F}(\mathcal{W}_j).`

.. literalinclude:: ../examples/autonomous_car_road.py
   :start-after: @setF@
   :end-before: @setF_end@

Now, we can validate whether the RHTLPProb object satisfies all the sufficient conditions.

.. literalinclude:: ../examples/autonomous_car_road.py
   :start-after: @valid@
   :end-before: @valid_end@

The result of the above ``validate()`` call is that the state in which
:math:`X_i = 0, \forall i \in \{0, \ldots, 29\}` is not in any :math:`\mathcal{W}_j,` 
i.e., the sufficient condition that the union of all :math:`\mathcal{W}_j`
covers the entire state space is not satisfied.
Since we know that we don't have to deal with the above state, we will exclude it.

.. literalinclude:: ../examples/autonomous_car_road.py
   :start-after: @exclude@
   :end-before: @exclude_end@
