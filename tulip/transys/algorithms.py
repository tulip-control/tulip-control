# Copyright (c) 2013-2014 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
"""
Algorithms on Kripke structures and Automata
"""
import logging
import copy
import warnings

from .transys import FiniteTransitionSystem
from .automata import BuchiAutomaton
from .mathset import MathSet

_hl = 40 *'-'

logger = logging.getLogger(__name__)

#TODO BA x BA sync prod algorithm

def ts_sync_prod(ts1, ts2):
    """Synchronous (tensor) product with other FTS.
    
    @type ts1, ts2: L{FiniteTransitionSystem}
    """
    
    prod_ts = FiniteTransitionSystem()
    
    # union of AP sets
    prod_ts.atomic_propositions |= \
        ts1.atomic_propositions | ts2.atomic_propositions
    
    # use more label sets, instead of this explicit approach
    #
    # for synchronous product: Cartesian product of action sets
    #prod_ts.actions |= ts1.actions * ts2.actions
    
    prod_ts = super(FiniteTransitionSystem, self).tensor_product(
        ts, prod_sys=prod_ts
    )
    
    return prod_ts

def sync_prod(ts, ba):
    """Synchronous product between (BA, TS), or (BA1, BA2).
    
    The result is always a L{BuchiAutomaton}:
    
        - If C{ts_or_ba} is a L{FiniteTransitionSystem} TS,
            then return the synchronous product BA * TS.
            
            The accepting states of BA * TS are those which
            project on accepting states of BA.
        
        - If C{ts_or_ba} is a L{BuchiAutomaton} BA2,
            then return the synchronous product BA * BA2.
    
            The accepting states of BA * BA2 are those which
            project on accepting states of both BA and BA2.
    
            This definition of accepting set extends
            Def.4.8, p.156 U{[BK08]
            <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
            to NBA.
    
    Synchronous product TS * BA or TS1 * TS2.
    
    Returns a Finite Transition System, because TS is
    the first term in the product.
    
    Changing term order, i.e., BA * TS, returns the
    synchronous product as a BA.
    
    Caution
    =======
    This method includes semantics for true\in\Sigma (p.916, U{[BK08]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}),
    so there is a slight overlap with logic grammar.  In other
    words, not completely isolated from logics.
    
    See Also
    ========
    L{transys._ts_ba_sync_prod}
    
    @param ts_or_ba: other with which to take synchronous product
    @type ts_or_ba: L{FiniteTransitionSystem} or L{BuchiAutomaton}
    
    @return: self * ts_or_ba
    @rtype: L{BuchiAutomaton}
    
    See Also
    ========
    __mul__, async_prod, BuchiAutomaton.sync_prod, tensor_product
    Def. 2.42, pp. 75--76 U{[BK08]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
    Def. 4.62, p.200 U{[BK08]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
    
    @param ts_or_ba: system with which to take synchronous product
    @type ts_or_ba: L{FiniteTransitionSystem} or L{BuchiAutomaton}
    
    @return: synchronous product C{self} x C{ts_or_ba}
    @rtype: L{FiniteTransitionSystem}
    """
    if not isinstance(ba, BuchiAutomaton):
        raise Exception
    if not isinstance(ts, FiniteTransitionSystem):
        raise Exception

def add(self, other):
    """Merge two Finite Transition Systems.
    
    States, Initial States, Actions, Atomic Propositions and
    State labels and Transitions of the second Transition System
    are merged into the first and take precedence, overwriting
    existing labeling.
    
    Example
    =======
    This can be useful to construct systems quickly by creating
    standard "pieces" using the functions: line_labeled_with,
    cycle_labeled_with
    
    >>> n = 4
    >>> L = n*['p'] # state labeling
    >>> ts1 = line_labeled_with(L, n-1)
    >>> ts1.plot()
    >>> 
    >>> L = n*['p']
    >>> ts2 = cycle_labeled_with(L)
    >>> ts2.states.add('s3', ap={'!p'})
    >>> ts2.plot()
    >>> 
    >>> ts3 = ts1 +ts2
    >>> ts3.transitions.add('s'+str(n-1), 's'+str(n) )
    >>> ts3.default_layout = 'circo'
    >>> ts3.plot()
    
    See Also
    ========
    L{line_labeled_with}, L{cycle_labeled_with}
    
    @param other: system to merge with
    @type other: C{FiniteTransitionSystem}
    
    @return: merge of C{self} with C{other}, union of states,
        initial states, atomic propositions, actions, edges and
        labelings, those of C{other} taking precedence over C{self}.
    @rtype: L{FiniteTransitionSystem}
    """
    if not isinstance(other, FiniteTransitionSystem):
        msg = 'other class must be FiniteTransitionSystem.\n'
        msg += 'Got instead:\n\t' +str(other)
        msg += '\nof type:\n\t' +str(type(other) )
        raise TypeError(msg)
    
    self.atomic_propositions |= other.atomic_propositions
    self.actions |= other.actions
    
    # add extra states & their labels
    for state, label in other.states.find():
        if state not in self.states:
            self.states.add(state)
        
        if label:
            self.states[state]['ap'] = label['ap']
    
    self.states.initial |= set(other.states.initial)
    
    # copy extra transitions (be careful w/ labeling)
    for (from_state, to_state, label_dict) in \
        other.transitions.find():
        # labeled edge ?
        if not label_dict:
            self.transitions.add(from_state, to_state)
        else:
            sublabel_value = label_dict['actions']
            self.transitions.add(
                from_state, to_state, actions=sublabel_value
            )
    
    return copy.copy(self)

def async_prod(self, ts):
    """Asynchronous product TS1 x TS2 between FT Systems.
    
    See Also
    ========
    __or__, sync_prod, cartesian_product
    Def. 2.18, p.38 U{[BK08]
    <http://tulip-control.sourceforge.net/doc/bibliography.html#bk08>}
    """
    if not isinstance(ts, FiniteTransitionSystem):
        raise TypeError('ts must be a FiniteTransitionSystem.')
    
    if self.states.mutants or ts.states.mutants:
        mutable = True
    else:
        mutable = False
    
    # union of AP sets
    prod_ts = FiniteTransitionSystem(mutable=mutable)
    prod_ts.atomic_propositions |= \
        self.atomic_propositions | ts.atomic_propositions
    
    # for parallel product: union of action sets
    prod_ts.actions |= self.actions | ts.actions
    
    prod_ts = super(FiniteTransitionSystem, self).cartesian_product(
        ts, prod_sys=prod_ts
    )
    
    return prod_ts

def load_spin2fts():
    """

    UNDER DEVELOPMENT; function signature may change without
    notice.  Calling will result in NotImplementedError.
    """
    raise NotImplementedError
