FMU export
==========

This directory contains the Python and C files used to export TuLiP controllers
as a Functional Mock-up Unit (FMU) for co-simulation.

Installation
------------

Just do it
``````````

In most cases, preparing the dependencies is achieved by (building ecos is not
necessary) ::

  cd ../../extern/ecos
  ./get.sh

If that fails, you may need to perform some steps manually and possibly make
adjustments.  Consult the next section.

Details
```````

Unless stated otherwise, paths are relative to the directory of this README,
i.e. ``contrib/fmu`` in the TuLiP source tree.

There are two notable external dependencies:

* `ECOS <https://github.com/embotech/ecos>`_ : Consult ``../../extern/ecos/README``.
* FMI headers : Copies of these are provided under ``../../extern/fmi``.

The FMI headers may be used as provided.  For the former dependency, a
particular release of ECOS must be obtained and patched.  This process is
automated by the ``get.sh`` shell script under ``../../extern/ecos``.  If it
fails, then that script together with the corresponding README file are enough
to complete the required steps manually.

It is *not* necessary to build ecos.  Building of ecos (and other items
distributed with it) is performed via the Makefile of the example.

As demonstrated in the example, it is possible to use the FMU in Ptolemy II,
which can be obtained from http://ptolemy.eecs.berkeley.edu/
However, Ptolemy II is not necessary for the export routine.

The following standard tools must also be installed:

* C compiler (e.g., `gcc`)
* ``make``
* ``zip``

Finally, the manner of building the FMU depends on the target platform.  This
can be indicated manually by defining the ARCH variable in the Makefile.  E.g.,
64-bit Mac OS can be forced by adding ``ARCH=darwin64`` at the beginning of
``Makefile``.  If undefined, then ``ARCH`` will be determined from ``uname``.


Example
-------

If the installation steps succeeded, then it should suffice to ::

  python robotfmu.py

The resulting FMU file is ``TuLiPFMU.fmu``.  If Ptolemy II is installed, then
run a simulation using the FMU ::

  vergil model.xml

A demo of the generated C code is given by ::

  build/test_controller

To remove all generated files except the FMU and any executables, ::

  make clean

To return to a pristine directory, use ``make purge``.


Organization
------------

The various parts of the FMU export code are intertwined with the example,
robotfmu.py.  As such, until it is integrated into TuLiP, the best way to begin
is by reading the documentation inside robotfmu.py and the following:

* ctrlexport.py : exporting the Mealy meachine
* pppexport.py : exporting the proposition preserving partition
* poly2str.py : miscellaneous functions for converting matrices and polytopes to
  C code
* exportFMU.py : generating FMU

Note that ``make`` is invoked from within robotfmu.py and exportFMU.py using
os.system() from the Python standard library.  The corresponding directives are
defined in ``Makefile``.
