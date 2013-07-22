'''
Created on Aug 13, 2012

@author: ewolff
'''
import numpy as np
import time
import ex_graph as gr
import random
import itertools
import networkx as nx


######################################################################
## Start of main function
######################################################################
ti = time.time()

# INPUT: TS, spec, prop2states, costs
tf_model = time.time()

TS_orig = TS.copy()
print spec

print "Original TS ",len(TS)#, TS


## Determine all nodes that are blocking, i.e., don't have any valid actions left
t0 = time.time()
blockingNodes = gr.blocking_nodes(TS)
t1 = time.time()
print "blockingNodes:",len(blockingNodes)
print t1-t0,"(sec)"


# GR1 fixpoint from papers



