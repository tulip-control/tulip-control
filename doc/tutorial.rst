Tutorial and Examples
=====================

TuLiP is developed for synthesis of discrete controllers for hybrid control
systems, including supervisory controllers, switching controllers and
receding horizon temporal logic planners (not supported in this version).

Synthesis of Reactive Controllers
---------------------------------
To illustrate the basic synthesis capabilities of TuLiP, we synthesize a
simple discrete state controller for a robotic motion control system.

Problem Formulation
```````````````````
We consider a system that comprises the physical component, which we refer
to as the plant, and the (potentially dynamic and not a priori known)
environment in which the plant operates.  The system may contain both
continuous (physical) and discrete (computational) components.  In summary,
the problem we are interested in consists of

  - discrete system state,
  - continuous system state,
  - (discrete) environment state, and
  - specification.

Here, `discrete` state refers to the state that can take only a finite number
of possible values while `continuous` state refers to the state that can take
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

     .. autofunction:: abstract.prop2part
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

The LtiSysDyn class is used to define continuous dynamics.

     .. autoclass:: hybrid.LtiSysDyn
	:noindex:

Once we have the proposition preserving partition and the continuous dynamics,
continuous state space discretization can be done using the following function call:

     .. autofunction:: abstract.discretize
	:noindex:

.. _ssec:syn:

Digital design synthesis
````````````````````````
The continuous state space discretization generates a finite state
abstraction of the continuous state, represented by a finite transition
system.  Each state in this finite transition system corresponds to a cell
in the continuous domain.  A transition :math:`c_i \to c_j` in this finite
state system indicates that from any continuous state :math:`s_0` that
belongs to cell :math:`c_i`, there exists a sequence of control inputs
:math:`u_0, u_1, \ldots, u_{N-1}` that takes the system to another
continuous state :math:`s_{N}` in cell :math:`c_j`.  Hence, under the
assumption that the specification is stutter invariant, we can describe the
continuous dynamics by an LTL formula of the form

.. math::
   (v = c_i) \implies next(\bigvee_{j \text{ s.t. } c_i \to c_j} v = c_j),

where :math:`v` is a new discrete variable that describes in which cell
the continuous state is.

Since the partition is proposition preserving, all the continuous states
that belong to the same cell satisfy exactly the same set of propositions on
the continuous state. By the abuse of notation, we write :math:`c_j \models
X_i` if all the continuous states in cell :math:`c_j` satisfy proposition
:math:`X_i`.  Then, we can replace any proposition :math:`X_i` on the
continuous state variables by the formula :math:`\displaystyle{\bigvee_{j
\text{ s.t. } c_j \models X_i} v = c_j}`.

Putting everything together, we now obtain a specification of the form in
:eq:`spec` (see also :doc:`specifications`).  We can then use the GR(1) game
implementation in `JTLV <http://jtlv.ysaar.net/>`_ or `gr1c
<http://scottman.net/2012/gr1c>`_ to automatically synthesize a planner that
ensures the satisfaction of the specification, taking into account all the
possible behaviors of the environment.  This is done using the
:literal:`synth.synthesize` function:

    .. autofunction:: synth.synthesize
	:noindex:

The resulting output is a finite state machine (Mealy machine):

    .. autofunction:: transys.FiniteStateMachine
	:noindex:

.. _ssec:ex1:

Example 1: Discrete State Robot Motion Planning
```````````````````````````````````````````````
This example is provided in examples/discrete.py.
It illustrates the use of the gr1c module in synthesizing a planner
for a robot that only needs to make discrete decision.

.. image:: robot_simple.*
   :align: center

We consider the robot moving around the regions as shown in the above figure
while receiving externally triggered park signal.
The specification of the robot is

.. math::
   \varphi = \square \diamond(\neg park) \implies (\square \diamond(s \in C_5)
   \wedge \square(park \implies \diamond(s \in C_0))).

We cannot, however, deal with this specification directly since it is not in
the form of GR(1).  An equivalent GR(1) specification of the above
specification can be obtained by introducing an auxiliary discrete system
variable :math:`X0reach,` initialized to `True`. The transition relation of
:math:`X0reach,` is given by :math:`\square(\text{next}(X0reach) = (s \in
C_0 \vee (X0reach \wedge \neg park))).`

To automatically synthesize a planner for this robot, we first import the
necessary modules:

.. highlight:: python

.. literalinclude:: ../examples/discrete.py
   :start-after: @import_section@
   :end-before: @import_section_end@

We next define the dynamics of the system, modeled as a discrete transition
system in which the robot can be located anyplace no a 2x3 grid of cells.
Transitions between adjacent cells are allowed, which we model as a
transition system in this example (it would also be possible to do this via
a formula):

.. literalinclude:: ../examples/discrete.py
   :start-after: @system_dynamics_section@
   :end-before: @system_dynamics_section_end@

To create the specification, we label some of the states with names:

.. literalinclude:: ../examples/discrete.py
   :start-after: @system_labels_section@
   :end-before: @system_labels_section_end@

These names serve as atomic propositions that are true when the system is in
the indicated states.

The environment can issue a park signal that requires the robot to respond
by moving to the lower left corner of the grid.  We assume that
the park signal is turned off infinitely often.  We describe this using the
following code:

.. literalinclude:: ../examples/discrete.py
   :start-after: @environ_section@
   :end-before: @environ_section_end@

Here the specification is broken up into four pieces: a description of the
discrete environment variables (:literal:`env_vars`), a specification for
the initial condition for the environment (:literal:`env_init`), a progress
formula (:literal:`env_prog`) that must be satisfied infinitely often, and
a safety formula (:literal:`env_safe`) that must hold at all times during
the execution.  The :literal:`set()` command is used to initialize one or
more of these variables to the empty set.

The system specification is that the robot should repeatedly revisit
the upper right corner of the grid while at the same time responding
to the park signal by visiting the lower left corner.  The LTL
specification is given by 

.. math::
   \square\diamond home \wedge \square (park \implies \diamond lot)

Since this specification is not in GR(1) form, we introduce the
variable X0reach that is initialized to True and the specification
:math:`\square(park \implies \diamond lot)` becomes

.. math::
     \square( (next(X0reach) = lot) \vee (X0reach \wedge \neg park))

The python code to implement this logic is given by:

.. literalinclude:: ../examples/discrete.py
   :start-after: @specs_setup_section@
   :end-before: @specs_setup_section_end@

Note the use of :literal:`<->` for equivalence (equality).  As in the case
of the environmental specification, the system specification consists of
four parts that provide additional discrete system variables
(:literal:`sys_vars`), initial conditions (:literal:`sys_init`), progress
conditions (:literal:`sys_prog`) and safety conditions
(:literal:`sys_safe`). 

Finally, we construct the full specification for the system and environment
by creating a GR(1) specification consisting of the various pieces we have
constructed:

.. literalinclude:: ../examples/discrete.py
   :start-after: @specs_create_section@
   :end-before: @specs_create_section_end@

To synthesize the controller, we call the :literal:`synth.synthesize()`
function. 

.. literalinclude:: ../examples/discrete.py
   :start-after: @synthesize@
   :end-before: @synthesize_end@

The controller can now be saved in graphical form, or printed if pydot package
is not available:

.. literalinclude:: ../examples/discrete.py
   :start-after: @plot_print@
   :end-before: @plot_print_end@

.. _ssec:ex2:

Example 2: Continuous State Robot Motion Planning
`````````````````````````````````````````````````
This example is provided in examples/continuous.py.
It is an extension of the previous example by including continuous dynamics.

First, we import the necessary modules, 
specify the smv file, spc file and aut file,
and specify the environment and the discrete system variables
as in the previous example.

.. literalinclude:: ../examples/continuous.py
   :start-after: @import_section@
   :end-before: @import_section_end@

Next, we specify the continuous dynamics.  This includes specifying the
continuous state space, propositions on continuous variables, and the
dynamics.  The robot dynamics in this case is :math:`\dot{x} = u_x + 5d_x, 
\dot{y} = u_y + 5d_y,` discretized with a sampling time of 0.2.

.. literalinclude:: ../examples/continuous.py
   :start-after: @dynamics_section@
   :end-before: @dynamics_section_end@

Now, we can construct the proposition preserving partition of the continuous
state space and discretize the continuous state space based on the dynamics.

.. literalinclude:: ../examples/continuous.py
   :start-after: @partition_section@
   :end-before: @partition_section_end@

.. literalinclude:: ../examples/continuous.py
   :start-after: @discretize_section@
   :end-before: @discretize_section_end@

The rest is the same as in the previous example.  We specify the
environment, create a GR(1) system specification, and synthesize a
controller.

.. literalinclude:: ../examples/continuous.py
   :start-after: @synthesize_section@
   :end-before: @synthesize_section_end@


Working with Systems with Piecewise Affine Dynamics
---------------------------------------------------
TuLiP can also handle piecewise affine dynamics of the form: 

for :math:`t \in \{0,1,2,...\}`

.. math::
   s[t+1]  &=   A_is[t] + B_iu[t] + E_id[t] + K_i\\
   u[t]    &\in U_i\\
   d[t]    &\in D_i\\
   s[t]	   &\in S_i
   :label: pwadynamics

where :math:`S_i \subseteq \mathbb{R}^n` for :math:`i \in \{0,1,2,
\ldots,n_s\}` form a polytopic partition of the state space :math:`S`, in
:eq:`dynamics`, :math:`U_i \subseteq \mathbb{R}^m` is the set of admissible
control inputs, :math:`D_i \subseteq \mathbb{R}^p` is the set of exogenous
disturbances within :math:`S_i`, and :math:`s[t], u[t], d[t]` are the
continuous state, the control signal and the exogenous disturbance,
respectively, at time :math:`t`.

LtiSysDyn class is used to represent subsystems of the form
:eq:`pwadynamics`.

     .. autoclass:: hybrid.LtiSysDyn
	:noindex:

The subsystems can be put together to define a piecewise affine system which
is represented by PwaSysDyn class.

     .. autoclass:: hybrid.PwaSysDyn
	:noindex:


Example 3: Robot Motion Planning with Piecewise Affine Dynamics
```````````````````````````````````````````````````````````````

This example is provided in examples/pwa.py.
It is an extension of the previous examples including a robot model
with piecewise affine dynamics.

Assume our robot is traveling on a nonhomogenous surface (x-y plane), 
resulting in different dynamics at different parts of the plane. 
Since the continuous state space in this example is just x-y position, different
dynamics in different parts of the surface can be modeled as a piecewise 
affine system. When :math:`s[t] \in[0, 3]\times[0.5, 2]`, the following dynamics
are active:

.. literalinclude:: ../examples/pwa.py
   :start-after: @subsystem0@
   :end-before: @subsystem0_end@

When :math:`s[t] \in[0, 3]\times[0, 0.5]`, the following dynamics
are active:

.. literalinclude:: ../examples/pwa.py
   :start-after: @subsystem1@
   :end-before: @subsystem1_end@

Piecewise affine system can be formed from the dynamics of its subsystems.

.. literalinclude:: ../examples/pwa.py
   :start-after: @pwasystem@
   :end-before: @pwasystem_end@

Discretization and synthesis follow exactly as before.



