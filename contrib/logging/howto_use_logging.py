#!/usr/bin/env python

"""
This example shows how to load from a json file
a configuration for logging messages from tulip.

You can configure it also directly in python, but
rewriting or copy/pasting again and again the same
base configuration is not very efficient.
"""
import logging.config
import json
import pprint

fname = 'logging_config.json'
with open(fname, 'r') as f:
    config = json.load(f)
print('Your logging config is:\n{s}')
pprint.pprint(config)

logging.config.dictConfig(config)

from tulip import transys, spec, synth

sys = transys.FTS()
sys.states.add_from({0, 1})
sys.states.initial.add(0)

sys.add_edges_from([(0, 1), (1, 0)])

sys.atomic_propositions.add('p')
sys.node[0]['ap'] = {'p'}

specs = spec.GRSpec(sys_vars={'p'}, sys_prog={'p'})

mealy = synth.synthesize('gr1c', specs, sys=sys)
