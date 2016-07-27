--- Version 0.1a released ----

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
