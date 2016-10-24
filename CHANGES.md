# tulip changelog


## 1.3.0
2016-11-18

- support synthesis of:

  - both Moore and Mealy strategies
  - non-circular Klein-Pnueli-style specifications
  - options to select quantification of initial values of variables

- introduce alternatives for definition of synthesis, defined by
  attributes of `tulip.spec.form.GRSpec`:

  - `moore`: synthesize a Moore strategy
  - `qinit`: select quantification of initial conditions
    (existential or universal for environment or system variables)
  - `plus_one`: non-circular specifications

- use `omega.games.gr1` as default GR(1) game solver
  This change allows a working basic installation in Python

- `gr1c` is not required. Now it is optional.

- update examples to demonstrate new synthesis options

- re-introduce and improve the module `tulip.gridworlds`

- implement feedback control with 1 and inf norms using `scipy`,
  thus making `tulip` independent of `cvxopt`.

- add function `interfaces.print_env` to help users
  inspect their environment, for easier installation and maintenance

- update user documentation, installation instructions

- require `omega >= 0.0.9, <= 0.1.0`

- bump requirement to `pydot >= 1.2.0`

- run all tests on
  [Travis CI](https://travis-ci.org/tulip-control/tulip-control/)

- fix bug and change continuous reachability computation,
  and effect of the option `use_all_horizon`

- remove args `bool_states, bool_actions` to function `tulip.synth.synthesize`

- remove option `init_option` from `interfaces.gr1c` and `gr1py`

- enumerate strategy in `interfaces.omega` using search

- add function `interfaces.omega.is_realizable`


## 1.2.1
2016-07-25

Version 1.2.1 of TuLiP is available on the Python Package Index (PyPI) at
https://pypi.python.org/pypi/tulip/1.2.1

Summary of changes for this release:

- add support for `gr1py`, an enumerative GR(1) solver in pure Python

- add support for `omega`,
  a symbolic GR(1) and Rabin(1) solver in pure Python,
  with optional Cython bindings to the C library CUDD

- modernize `setup.py`

- ease fully functional installation, as in `pip install tulip`

- modify `git` versioning, use `gitpython`

- relocate examples (shallower, no `robot_planning`)

- all core tests running on Travis CI

- require `pydot >= 1.1.0`, now available on PyPI

- remove `d3` support (experimental) from `transys`

- require `polytope >= 0.1.2`, its slower variant depends only on `scipy`

There were also several corrections (bug fixes).



## 1.2.0
2015-10-25

TuLiP is listed on the Python Package Index (PyPI) at
https://pypi.python.org/pypi/tulip

There have been many changes since the previous release, including:

- improvements to documentation;

- `cvxopt` and polytope are now optional dependencies;

- organization of tests into families to facilitate testing according to
optional dependencies that may be installed;
- `gr1c` is the default GR(1) solver;

- the `jtlv`-based GR(1) solver is not included by default, but a script
is provided to get it for users who want it.

- the addition of interfaces for new synthesis tools, such as slugs and
lily;

- (following previous item) the ability to synthesize for LTL
specifications;

- routines for converting several templates of formulae, including those
commonly referred to as "stability", "response", "eventuality", to GR(1).

Thanks to Ioannis Filippidis, Necmiye Ozay, Vasumathi Raman, and Richard
M. Murray for authoring and contributing to this release. Also thanks to
others who have provided feedback.


## 1.1a
2014-12-09

There has not been a release in a while, yet the project has been active.
We collect here several of the most significant changes,
though be warned there is more.
A good place to begin is the installation instructions at
http://tulip-control.sourceforge.net/doc/install.html
or fetch the release from
http://sourceforge.net/projects/tulip-control/files/

 - `tulip.polytope` is now a separate Python package,
   which is named simply `polytope`. The primary repository for it is https://github.com/tulip-control/polytope
   and releases are available from PyPI at
   https://pypi.python.org/pypi/polytope

- support for switched systems with external inputs is introduced

- multiprocessing can be used for discretization of switched systems

- redesign of abstraction (`AbstractPwa`, `AbstractSwitched`) and
  partition classes (`PropPreservingPartition` and
  parents that now live in `polytope`)

- `PropPreservingPartition` can now check several of its properties

- support for open finite transition systems in synth

- support for arbitrary finite domains when using the `gr1c` solver

- multiple system and environment actions supported

- `MealyMachine` now has an Sinit special initial state

- replaced `pyparsing` by `ply` as the default LTL parse generator
  (orders of magnitude faster)

- overhauled `transys` simplifying it resemble `networkx` more,
  introducing `LabeledDiGraph` and `TypedDict`

- more graphics functions, including:
  animation of discretization
  projection of Mealy strategies on partitions of the state space
  `PwaSysDyn` with their polytopic partition and (nonsmooth) vector field

- installation script and instructions for *nix machines in contrib

- instrumented code with logging

The next major planned release is version 1.2.0.
It will include changes that will break compatibility with this release.
From version 1.2.0 onward,
greater care will be taken to ensure backwards compatibility and
to clearly indicate parts of the API that are not stable.


## 0.1a
2012-08-22

2011-04-19  Richard Murray  <murray@malabar.local>

- `setup.py`: Added matlab files to setup, to make sure they get
  installed in the proper location

- `tulip/discretizeM.py` (discretizeM): changed path to reflect new
  location of `matlab/` (as a subdir of `tulip/`)

2011-04-19  Richard Murray  <murray@malabar.local>

- `tulip/discretizeM.py` (discretizeM): updated path for matlab files
	from `..` to `../matlab` (reflecting new source code location)

- `examples/*.py`: updated path for tulip files from `..` to `../tulip`

- `matlab/runDiscretizeMatlab.m` (HK): add initialization of variable
  HK to stop MATLAB 2010a error message
