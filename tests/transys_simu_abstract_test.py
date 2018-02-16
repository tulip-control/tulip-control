#!/usr/bin/env python
"""
Tests for transys.transys.simu_abstract (part of transys subpackage)
"""

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

from tulip.transys import transys as trs
from trs import simu_abstract


def simu_abstract_test():
    
    
    # build test FTS
    test_FTS = FTS()
    test_FTS.add_states