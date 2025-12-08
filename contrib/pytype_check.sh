# Type checking of `tulip` using `pytype`.
# Assumes that `tulip` dependencies have been installed.


# Install dependencies from PyPI
pip install --upgrade \
    pip \
    setuptools \
    wheel
# Install analysis tools
pip install --upgrade \
    pytype
# Statically analyze `tulip`
pytype --tree tulip
pytype --unresolved tulip
pytype \
    -v 1 \
    -k \
    -j 'auto' \
        tulip \
        setup.py \
        run_tests.py \
    -x tulip/interfaces/stormpy.py
