# tulip changelog


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
