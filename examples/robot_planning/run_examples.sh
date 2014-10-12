#!/usr/bin/env bash

# run all examples and stop at first error
set -e

python discrete.py
python gr1.py
python gr1_set.py
python hybrid.py
python only_mode_controlled.py
python environment_switching.py
python continuous.py
python pwa.py
