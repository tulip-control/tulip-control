Notes on Implementation
=======================

Notes here are given about the implementation for future developers, since this
extension is not documented with the rest of Tulip. Hopefully, this will help
anyone who maintains this interface if I am ever unable to.

-Stephanie Tsuei, June 2014



Importing to Stateflow
----------------------

This section documents the quirks of importing a Tulip MealyMachine to a
Stateflow MealyMachine.

- All Simulink signals must be finite numbers. (No strings, `Inf`, or `NaN`
  allowed.) Inputs to Stateflow charts must be Simulink signals. However,
  `env_action` and `sys_action` inputs in Tulip must be strings.

  To get around this, while importing to Stateflow, a different integer is
  generated for each input and output and stored in hash tables called
  `input_value_map` and `output_value_map` that map the original strings to
  these randomly generated numbers. These randomly generated numbers are used in
  place of the strings in Simulink/Stateflow.


- Initial state problems:

  Tulip MealyMachine objects have a single initial state called `Sinit`. The
  outgoing transitions from `Sinit` specify the initial environment and system
  modes. The `loc` variable in the state to which we can transition to from
  `Sinit` is the initial location of the system. (Each location is a cell in the
  abstraction.) These "initial transitions" would not be part of a physical
  simulation, but can be simulated manually in Python.

  On the other hand, the transitions described above all must occur before a
  Simulink model starts simulating. Simulink will not simulate any initial
  states. Therefore, if we have the transition

                  `Sinit -> A -> B`

  the transition that needs to be the initial transition in Simulink is

                  `A -> B` (^)

  To work around this quirk,

    1. The initial location is an input into the Stateflow chart. The only
    transitions that depend on the initial location are transitions like (^).
    Then, with the current environment mode as a second input, there should be
    only one valid initial transition for any given simulation, and Simulink
    will choose the right one. (Simulink models allow for multiple initial
    transitions.)


  Note: In Simulink, initial transitions are known as "default transitions".



Receding Horizon Control
------------------------

- When a system is discretized with `closed_loop = False`, receding horizon
  inputs are computed every time the Stateflow chart moves. The horizon used to
  compute the receding horizon input is implemented as a constant block in
  Simulink.

- When a system is discretized with `closed_loop = True` and `N` timesteps,
  (and the Stateflow chart has a timestep of `T`) receding horizon inputs are
  computed every `T/N` seconds with a decreasing horizon each time. The horizon
  used to compute the receding horizon input is implemented as its own Stateflow
  Mealy Machine.



Structure of `regions`
----------------------

Every entry of the struct array `regions` has two fields:

    - `index`: the location number of the region in the Python abstraction
      object
    - `region`: A `Polyhedron` (class defined in MPT).



Structure of `MPTsys`
---------------------

If the problem does not involve a `SwitchedSysDyn` object, then the `MPTsys`
object is a `LTISystem` or a `PWASystem` (both defined in MPT). Otherwise,
`MPTsys` is a struct array where each entry has the fields

    - `sys_action`
    - `env_action`
    - `system` - a `PWASystem`
