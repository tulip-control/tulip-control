Class: MDP (and UncertainMDP)
Author: James Bern

Contact: jbern@caltech.edu

Date: 2013-Aug-23

0: Organization
MDP has child UncertainMDP, defined in MDP_overhaul and uncertain_MDP_overhaul resp.
probability classes are defined in probability
MDP_functions holds accessory functions that work on both MDP and UncertainMDP
value_iteration_function holds functions specific to value iteration

I. Examples
Testing was done by examples, which are left as functions in the files, and
are called by the main function.  To run the examples for a particular class,
run the class in the Terminal.  For example, python value_iteration.py will
display the examples for value_iteration.  It is encouraged to use the examples
to retest code after making modifications.  To this end, the hand-checked
output of each example is located in the main function of each file as comments.
Notation for actions roughly follows "[s_i:]s_f1/s_f2/...", and should be
intuitive.

II. Variable Conventions

Note: When possible I followed the convention of the paper or book
(e.g., Principles of Model Checking by Baier and Katoen) I was working
off of.  When not working off of an established algorithm I try to follow:

s_i: initial state when transitioning
s_f: final state when transitioning
a: action
p: probability

mdp: Markov decision process
prod: product Markov decision process
s: state in original mdp when important to differentiate
sq: state in product mdp when important to differentiate

ra: Rabin Automaton
L: bad half of acceptance pair
K: good half of acceptance pair

MEC: maximal end component
AMEC: accepting maximal end component
Note: plural as MECs and AMECs

spec: specification
foo_bar_tup: (foo, bar)

III. Overview

The bulk of the MDP class is the T field, which represents transitions.
form: T[s_i][a] = list of destination tuples (s_f, p).
The C field is identical in form and is C[s_i][a] = float of cost.

The rest of the class is a set of useful functions, including one-step pre's,
one step post, and a few less typical ones:

*destinations is a one step post under a particular action

*condense_split_dynamics "cleans up" split probabilities (see docstring).
 Note: when generating a product MDP, condense_splity_dynamics should always
 be run on the initial MDP.  If it is not, the check of determinism
 (no action has destinations with duplicate s states) may fail.

*sanity_check provides a few simple checks on the MDP's structure, and
 best practice is to run this function after creating a new MDP or
 modifying a pre-existing one. 

The UncertainMDP class is currently just an extension of the MDP class with:
*IntervalProbabilities rather than float probabilities.
*a different sanity_check (see code) 

IV. Limitations/Issues

*The policies at the end of robot_MDP and robot_uncertain_MDP are largely untested.
 One specific thing to do, is to more carefully calculate the epsilon value
 for equality of value in gen_actMax (currently I'm using a relatively
 unconservative placeholder of .001).  This magic number should be replaced
 with something based on the epsilon value used in value iteration.

*The RabinAutomaton used in product generation is currently hand-coded but
 eventually should be read in from ltl2dstar.  More pressingly the RabinAutomaton
 used in the code is not current with TuLiP, and must be replaced with the
 current version.  Also, LK_lst should be an attribute of the RabinAutomaton
 (whether LK_lst is still passed to functions, or the RabinAutomaton is passed
 instead does not seem particularly important.)

*The 'satisfies' sub-function of the generate_product_MDP function has strict
 restrictions on the form of the label that's passed to it.  The sub-function 'satisfies' should
 be generalized as needed, and some assertion/warning should be added.

*induce_digraph should be replaced entirely by induce_nx_digraph, but currently
 is used in max_end_components.  Something seems buggy with networkx's Graph
 construction from a dict_of_lists.

*The states field is awkward can cause errors if not treated carefully.
 It might be better replaced with def states(self): return self.T.keys().
 Alternatively, it could be replaced with the rich States object in TuLiP.

V. Future Work

*Policies should be formalized, and generalized into functions.

*More types of transition uncertainty sets should be implemented for the UncertainMDP class.

*Richer simulation with plotting should be added.
