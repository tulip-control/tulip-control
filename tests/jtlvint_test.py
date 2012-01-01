#!/usr/bin/env python
"""
Based on test code moved here from bottom of tulip/jtlvint.py

SCL; 31 Dec 2011.
"""

import os

from tulip.jtlvint import *
from tulip.polytope import Region
from tulip.prop2part import PropPreservingPartition


class jtlvint_test:

    def setUp(self):
        self.tmpdir = "tmpspec"
        testfile = os.path.join(self.tmpdir, "testjtlvint")
        self.smvfile = testfile + '.smv'
        self.spcfile = testfile + '.spc'
        self.autfile = testfile + '.aut'
    #    env_vars = {'park' : 'boolean', 'cellID' : '{0,...,3,4,5}'}
        env_vars = {'park' : 'boolean', 'cellID' : [0,1,2,3,4,5]}
        sys_disc_vars = {'gear' : '{-1...1}'}
        cont_props = ['X0', 'X1', 'X2', 'X3', 'X4']
        disc_dynamics = PropPreservingPartition()
        region0 = Region('p0', [1, 0, 0, 0, 0])
        region1 = Region('p1', [0, 1, 0, 0, 0])
        region2 = Region('p2', [1, 0, 1, 0, 0])
        region3 = Region('p3', [0, 0, 0, 1, 0])
        region4 = Region('p4', [0, 0, 0, 0, 1])
        region5 = Region('p5', [1, 0, 0, 1, 1])
        disc_dynamics.list_region = [region0, region1, region2,
                                     region3, region4, region5]
        disc_dynamics.num_regions = len(disc_dynamics.list_region)
        disc_dynamics.trans = [[1, 1, 0, 1, 0, 0],
                               [1, 1, 1, 0, 1, 0],
                               [0, 1, 1, 0, 0, 1],
                               [1, 0, 0, 1, 1, 0],
                               [0, 1, 0, 1, 1, 1],
                               [0, 0, 1, 0, 1, 1]]
        disc_dynamics.list_prop_symbol = cont_props
        disc_dynamics.num_prop = len(disc_dynamics.list_prop_symbol)
        disc_props = {'Park' : 'park',
                      'X0d' : 'cellID=0',
                      'X1d' : 'cellID=1',
                      'X2d' : 'cellID=2',
                      'X3d' : 'cellID=3',
                      'X4d' : 'cellID=4',
                      'X5d' : 'gear = 1'}

        assumption = '[]<>(!park) & []<>(!X4d)'
        guarantee = '[]<>(X4d -> X4) & []<>X1 & [](Park -> X0)'
        spec = [assumption, guarantee]

        disc_dynamics=PropPreservingPartition()
        spec[1] = '[]<>(X0d -> X5d)'  
        newvarname = generateJTLVInput(env_vars=env_vars,
                                       sys_disc_vars=sys_disc_vars,
                                       spec=spec, disc_props=disc_props,
                                       disc_dynamics=disc_dynamics,
                                       smv_file=self.smvfile,
                                       spc_file=self.spcfile,
                                       verbose=2)

    def tearDown(self):
        os.remove(self.smvfile)
        os.remove(self.spcfile)
        os.remove(self.autfile)
        os.rmdir(self.tmpdir)
        
    def test_checkRealizability(self):
        assert checkRealizability(smv_file=self.smvfile,
                                  spc_file=self.spcfile,
                                  aut_file=self.autfile,
                                  heap_size='-Xmx128m',
                                  pick_sys_init=False,
                                  file_exist_option='r', verbose=3)

    def test_computeStrategy(self):
        assert computeStrategy(smv_file=self.smvfile,
                               spc_file=self.spcfile,
                               aut_file=self.autfile,
                               heap_size='-Xmx128m',
                               priority_kind='ZYX',
                               init_option=1,
                               file_exist_option='r', verbose=3)

        # assert computeStrategy(smv_file=self.smvfile,
        #                        spc_file=self.spcfile,
        #                        aut_file=self.autfile,
        #                        heap_size='-Xmx128m',
        #                        priority_kind='ZYX',
        #                        init_option=2,
        #                        file_exist_option='r', verbose=3)

        # assert not computeStrategy(smv_file=self.smvfile,
        #                            spc_file=self.spcfile,
        #                            aut_file=self.autfile,
        #                            heap_size='-Xmx128m',
        #                            priority_kind='ZYX',
        #                            init_option=0,
        #                            file_exist_option='r', verbose=3)

    # def test_synthesize(self):
    #     assert synthesize(env_vars=env_vars,
    #                       sys_disc_vars=sys_disc_vars,
    #                       spec=spec, disc_props=disc_props,
    #                       disc_dynamics=disc_dynamics,
    #                       smv_file=self.smvfile,
    #                       spc_file=self.spcfile,
    #                       aut_file=self.autfile,
    #                       heap_size='-Xmx128m', priority_kind=3,
    #                       init_option=init_option,
    #                       file_exist_option='r', verbose=3)

    # def test_getCounterExamples(self):
    #     counter_examples = getCounterExamples(aut_file=self.autfile, verbose=1)
