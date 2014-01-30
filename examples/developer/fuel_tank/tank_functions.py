"""
Auxiliary discretization and specification functions
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
from scipy import sparse as sp

from tulip import abstract
import tulip.polytope as pc
from tulip import transys as trs


