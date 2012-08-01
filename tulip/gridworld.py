# Copyright (c) 2012 by California Institute of Technology
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
# 
# $Id$
"""
Routines for working with gridworlds.

Note (24 June 2012): Several pieces of source code are taken or
derived from btsynth; see http://scottman.net/2012/btsynth
"""

import itertools
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

from polytope import Polytope, Region
from prop2part import prop2part2, PropPreservingPartition
from spec import GRSpec


class GridWorld:
    def __init__(self, gw_desc=None, prefix="Y"):
        """Load gridworld described in given string, or make empty instance.

        @param gw_desc: String containing a gridworld description, or
                        None to create an empty instance.
        @param prefix: String to be used as prefix for naming
                       gridworld cell variables.
        """
        if gw_desc is not None:
            self.loads(gw_desc)
        else:
            self.W = None
            self.init_list = []
            self.goal_list = []
        self.prefix = prefix
        self.offset = (0, 0)

    def __eq__(self, other):
        """Test for equality.

        Does not compare prefixes of cell variable names.
        """
        if self.W is None and other.W is None:
            return True
        if self.W is None or other.W is None:
            return False  # Only one of the two is undefined.
        if np.all(self.W != other.W):
            return False
        if self.goal_list != other.goal_list:
            return False
        if self.init_list != other.init_list:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.pretty(show_grid=True)

    def __getitem__(self, key):
        """Return variable name corresponding to this cell.

        Supports negative wrapping, e.g., if Y is an instance of
        GridWorld, then Y[-1,-1] will return the variable name of the
        cell in the bottom-right corner, Y[0,-1] the name of the
        top-right corner cell, etc.  As usual in Python, you can only
        wrap around once.
        """
        if self.W is None:
            raise ValueError("Gridworld is empty; no names available.")
        if len(key) != len(self.W.shape):
            raise ValueError("malformed gridworld key.")
        if key[0] < -self.W.shape[0] or key[1] < -self.W.shape[1] or key[0] >= self.W.shape[0] or key[1] >= self.W.shape[1]:
            raise ValueError("gridworld key is out of bounds.")
        if key[0] < 0:
            key = (self.W.shape[0]+key[0], key[1])
        if key[1] < 0:
            key = (key[0], self.W.shape[1]+key[1])
        return str(self.prefix)+"_"+str(key[0] + self.offset[0])+"_"+str(key[1] + self.offset[1])


    def state(self, key, offset=(0, 0)):
        """Return dictionary form of state with keys of variable names.

        Supports negative indices for key, e.g., as in __getitem__.

        The offset argument is motivated by the use-case of multiple
        agents whose moves are governed by separate "gridworlds" but
        who interact in a shared space; with an offset, we can make
        "sub-gridworlds" and enforce rules like mutual exclusion.
        """
        if self.W is None:
            raise ValueError("Gridworld is empty; no cells exist.")
        if len(key) != len(self.W.shape):
            raise ValueError("malformed gridworld key.")
        if key[0] < -self.W.shape[0] or key[1] < -self.W.shape[1] or key[0] >= self.W.shape[0] or key[1] >= self.W.shape[1]:
            raise ValueError("gridworld key is out of bounds.")
        if key[0] < 0:
            key = (self.W.shape[0]+key[0], key[1])
        if key[1] < 0:
            key = (key[0], self.W.shape[1]+key[1])
        output = dict()
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                output[self.prefix+"_"+str(i+offset[0])+"_"+str(j+offset[1])] = 0
        output[self.prefix+"_"+str(key[0]+offset[0])+"_"+str(key[1]+offset[1])] = 1
        return output


    def isEmpty(self, coord):
        """Is cell at coord empty?

        @param coord: (row, column) pair; supports negative indices.
        """
        if self.W is None:
            raise ValueError("Gridworld is empty; no cells exist.")
        if len(coord) != len(self.W.shape):
            raise ValueError("malformed gridworld coord.")
        if self.W[coord[0]][coord[1]] == 0:
            return True
        else:
            return False

    def setOccupied(self, coord):
        """Mark cell at coord as statically (permanently) occupied."""
        if self.W is None:
            raise ValueError("Gridworld is empty; no cells exist.")
        self.W[coord[0]][coord[1]] = 1

    def setEmpty(self, coord):
        """Mark cell at coord as empty."""
        if self.W is None:
            raise ValueError("Gridworld is empty; no cells exist.")
        self.W[coord[0]][coord[1]] = 0

    def plot(self, font_pt=18, show_grid=False, grid_width=2):
        """Draw figure depicting this gridworld.

        Figure legend:
          - "I" : possible initial position,
          - "G" : goal.

        @param font_pt: size (in points) for rendering text in the figure.
        """
        W = self.W.copy()
        W = np.ones(shape=W.shape) - W
        plt.imshow(W, cmap=mpl_cm.gray, aspect="equal", interpolation="nearest")
        if show_grid:
            xmin, xmax, ymin, ymax = plt.axis()
            x_steps = np.linspace(xmin, xmax, W.shape[1]+1)
            y_steps = np.linspace(ymin, ymax, W.shape[0]+1)
            for k in x_steps:
                plt.plot([k, k], [ymin, ymax], 'k-', linewidth=grid_width)
            for k in y_steps:
                plt.plot([xmin, xmax], [k, k], 'k-', linewidth=grid_width)
            plt.axis([xmin, xmax, ymin, ymax])
        for p in self.init_list:
            plt.text(p[1], p[0], "I", size=font_pt)
        for n,p in enumerate(self.goal_list):
            plt.text(p[1], p[0], "G" + str(n), size=font_pt)
        
    def pretty(self, show_grid=False, line_prefix="", path=[], goal_order=False):
        """Return pretty-for-printing string.

        @param show_grid: If True, then grid the pretty world and show
                          row and column labels along the outer edges.
        @param line_prefix: prefix each line with this string.
        """
        compress = lambda p: [ p[n] for n in range(len(p)-1) if p[n] != p[n+1] ]
        # See comments in code for the method loads regarding values in W
        if self.W is None:
            return ""
        
        # LEGEND:
        #  * - wall (as used in original world matrix definition);
        #  G - goal location;
        #  I - possible initial location.
        out_str = line_prefix
        def direct(c1, c2):
            (y1, x1) = c1
            (y2, x2) = c2
            if x1 > x2:
                return "<"
            elif x1 < x2:
                return ">"
            elif y1 > y2:
                return "^"
            elif y1 < y2:
                return "v"
            else: # c1 == c2
                return "."
        if show_grid:
            out_str += "  " + "".join([str(k).rjust(2) for k in range(self.W.shape[1])]) + "\n"
        else:
            out_str += "-"*(self.W.shape[1]+2) + "\n"
        #if path:
        #    path = compress(path)
        for i in range(self.W.shape[0]):
            out_str += line_prefix
            if show_grid:
                out_str += "  " + "-"*(self.W.shape[1]*2+1) + "\n"
                out_str += line_prefix
                out_str += str(i).rjust(2)
            else:
                out_str += "|"
            for j in range(self.W.shape[1]):
                if show_grid:
                    out_str += "|"
                if self.W[i][j] == 0:
                    if (i,j) in self.init_list:
                        out_str += "I"
                    elif (i,j) in self.goal_list:
                        if goal_order:
                            out_str += str(self.goal_list.index((i,j)))
                        else:
                            out_str += "G"
                    elif (i,j) in path:
                        out_str += direct((i,j), path[(path.index((i,j))+1) % len(path)])
                    else:
                        out_str += " "
                elif self.W[i][j] == 1:
                    out_str += "*"
                else:
                    raise ValueError("Unrecognized internal world W encoding.")
            out_str += "|\n"
        out_str += line_prefix
        if show_grid:
            out_str += "  " + "-"*(self.W.shape[1]*2+1) + "\n"
        else:
            out_str += "-"*(self.W.shape[1]+2) + "\n"
        return out_str

    def size(self):
        """Return size of gridworld as a tuple in row-major order."""
        if self.W is None:
            return (0, 0)
        else:
            return self.W.shape

    def loads(self, gw_desc):
        """Reincarnate using given gridworld description string.
        
        @param gw_desc: String containing a gridworld description.

        In a gridworld description, any line beginning with # is
        ignored (regarded as a comment). The first non-blank and
        non-comment line must give the grid size as two positive
        integers separated by whitespace, with the first being the
        number of rows and the second the number of columns.

        Each line after the size line is used to construct a row of
        the gridworld. These are read in order with maximum number of
        lines being the number of rows in the gridworld.  A row
        definition is whitespace-sensitive up to the number of columns
        (any characters beyond the column count are ignored, so in
        particular trailing whitespace is allowed) and can include the
        following symbols:

          - C{ } : an empty cell,
          - C{*} : a statically occupied cell,
          - C{I} : possible initial cell,
          - C{G} : goal cell (must be visited infinitely often).

        If the end of file is reached before all rows have been
        constructed, then the remaining rows are assumed to be empty.
        After all rows have been constructed, the remainder of the
        file is ignored.
        """
        ###################################################
        # Internal format notes:
        #
        # W is a matrix of integers with the same shape as the
        # gridworld.  Each element has value indicating properties of
        # the corresponding cell, according the following key.
        #
        # 0 - empty,
        # 1 - statically (permanently) occupied.
        ###################################################
        W = None
        init_list = []
        goal_list = []
        row_index = -1
        for line in gw_desc.splitlines():
            if row_index != -1:
                # Size has been read, so we are processing row definitions
                if row_index >= W.shape[0]:
                    break
                for j in range(min(len(line), W.shape[1])):
                    if line[j] == " ":
                        W[row_index][j] = 0
                    elif line[j] == "*":
                        W[row_index][j] = 1
                    elif line[j] == "I":
                        init_list.append((row_index, j))
                    elif line[j] == "G":
                        goal_list.append((row_index, j))
                    else:
                        raise ValueError("unrecognized row symbol \""+str(line[j])+"\".")
                row_index += 1
            else:
                # Still looking for gridworld size in the given string
                if len(line.strip()) == 0 or line.lstrip()[0] == "#":
                    continue  # Ignore blank and comment lines
                line_el = line.split()
                W = np.zeros((int(line_el[0]), int(line_el[1])),
                             dtype=np.int32)
                row_index = 0

        if W is None:
            raise ValueError("malformed gridworld description.")

        # Arrived here without errors, so actually reincarnate
        self.W = W
        self.init_list = init_list
        self.goal_list = goal_list


    def load(self, gw_file):
        """Read description from given file.

        Merely a convenience wrapper for the L{loads} method.
        """
        with open(gw_file, "r") as f:
            self.loads(f.read())

    def dumps(self, line_prefix=""):
        """Dump gridworld description string.

        @param line_prefix: prefix each line with this string.
        """
        if self.W is None:
            raise ValueError("Gridworld does not exist.")
        out_str = line_prefix+" ".join([str(i) for i in self.W.shape])+"\n"
        for i in range(self.W.shape[0]):
            out_str += line_prefix
            for j in range(self.W.shape[1]):
                if self.W[i][j] == 0:
                    if (i,j) in self.init_list:
                        out_str += "I"
                    elif (i,j) in self.goal_list:
                        out_str += "G"
                    else:
                        out_str += " "
                elif self.W[i][j] == 1:
                    out_str += "*"
                else:
                    raise ValueError("Unrecognized internal world W encoding.")
            out_str += "\n"
        return out_str


    def dumpsubworld(self, size, offset=(0, 0), prefix="Y"):
        """Generate new GridWorld instance from part of current one.

        Does not perform automatic truncation (to make desired
        subworld fit); instead a ValueError exception is raised.
        Possible initial positions and goals are not included in the
        returned GridWorld instance.

        @param size: (height, width)
        @param prefix: String to be used as prefix for naming
                       subgridworld cell variables.
        @rtype: L{GridWorld}
        """
        if self.W is None:
            raise ValueError("Gridworld does not exist.")
        if len(size) != len(self.W.shape) or len(offset) != len(self.W.shape):
            raise ValueError("malformed size or offset.")
        if offset[0] < 0 or offset[0] >= self.W.shape[0] or offset[1] < 0 or offset[1] >= self.W.shape[1]:
            raise ValueError("offset is out of bounds.")
        if size[0] < 1 or size[1] < 1 or offset[0]+size[0] > self.W.shape[0] or offset[1]+size[1] > self.W.shape[1]:
            raise ValueError("unworkable subworld size, given offset.")
        sub = GridWorld(prefix=prefix)
        sub.W = self.W[offset[0]:(offset[0]+size[0]), offset[1]:(offset[1]+size[1])].copy()
        return sub


    def dumpPPartition(self, side_lengths=(1., 1.), offset=(0., 0.)):
        """Return proposition-preserving partition from this gridworld.

        In setting the initial transition matrix, we assume the
        gridworld is 4-connected.

        @param side_lengths: pair (W, H) giving width and height of
                             each cell, assumed to be the same across
                             the grid.
        @param offset: 2-dimensional coordinate declaring where the
                       bottom-left corner of the gridworld should be
                       placed in the continuous space; default places
                       it at the origin.

        @rtype: L{PropPreservingPartition<prop2part.PropPreservingPartition>}
        """
        if self.W is None:
            raise ValueError("Gridworld does not exist.")
        domain = Polytope(A=np.array([[0,-1],
                                      [0,1],
                                      [-1,0],
                                      [1,0]], dtype=np.float64),
                          b=np.array([-offset[1],
                                       offset[1]+self.W.shape[0]*side_lengths[1],
                                       -offset[0],
                                       offset[0]+self.W.shape[1]*side_lengths[0]],
                                     dtype=np.float64))
        cells = {}
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                #adjacency[i]
                cells[self.prefix+"_"+str(i)+"_"+str(j)] \
                    = Polytope(A=np.array([[0,-1],
                                           [0,1],
                                           [-1,0],
                                           [1,0]], dtype=np.float64),
                               b=np.array([-offset[1]-(self.W.shape[0]-i-1)*side_lengths[1],
                                            offset[1]+(self.W.shape[0]-i)*side_lengths[1],
                                            -offset[0]-j*side_lengths[0],
                                            offset[0]+(j+1)*side_lengths[0]],
                                          dtype=np.float64))
        part = prop2part2(domain, cells)

        adjacency = np.zeros((self.W.shape[0]*self.W.shape[1], self.W.shape[0]*self.W.shape[1]), dtype=np.int8)
        for this_ind in range(len(part.list_region)):
            (prefix, i, j) = extract_coord(part.list_prop_symbol[part.list_region[this_ind].list_prop.index(1)])
            if self.W[i][j] != 0:
                continue  # Static obstacle cells are not traversable
            adjacency[this_ind, this_ind] = 1
            if i > 0 and self.W[i-1][j] == 0:
                symbol_ind = part.list_prop_symbol.index(prefix+"_"+str(i-1)+"_"+str(j))
                ind = 0
                while part.list_region[ind].list_prop[symbol_ind] == 0:
                    ind += 1
                adjacency[ind, this_ind] = 1
            if j > 0 and self.W[i][j-1] == 0:
                symbol_ind = part.list_prop_symbol.index(prefix+"_"+str(i)+"_"+str(j-1))
                ind = 0
                while part.list_region[ind].list_prop[symbol_ind] == 0:
                    ind += 1
                adjacency[ind, this_ind] = 1
            if i < self.W.shape[0]-1 and self.W[i+1][j] == 0:
                symbol_ind = part.list_prop_symbol.index(prefix+"_"+str(i+1)+"_"+str(j))
                ind = 0
                while part.list_region[ind].list_prop[symbol_ind] == 0:
                    ind += 1
                adjacency[ind, this_ind] = 1
            if j < self.W.shape[1]-1 and self.W[i][j+1] == 0:
                symbol_ind = part.list_prop_symbol.index(prefix+"_"+str(i)+"_"+str(j+1))
                ind = 0
                while part.list_region[ind].list_prop[symbol_ind] == 0:
                    ind += 1
                adjacency[ind, this_ind] = 1
        part.adj = adjacency
        return part
    
    def discreteTransitionSystem(self):
        """ Write a discrete transition system suitable for synthesis.
        Unlike dumpPPartition, this does not create polytopes; it is 
        nonetheless useful and computationally less expensive.
        
        @rtype: L{PropPreservingPartition<prop2part.PropPreservingPartition>}
        """
        disc_dynamics = PropPreservingPartition(list_region=[],
                            list_prop_symbol=[], trans=[])
        num_cells = self.W.shape[0] * self.W.shape[1]
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                flat = lambda x, y: x*self.W.shape[1] + y
                # Proposition
                prop = self[i,j]
                disc_dynamics.list_prop_symbol.append(prop)
                # Region
                r = [ 0 for x in range(0, num_cells) ]
                r[flat(i,j)] = 1
                disc_dynamics.list_region.append(Region("R_" + prop, r))
                # Transitions
                # trans[p][q] if q -> p
                t = [ 0 for x in range(0, num_cells) ]
                t[flat(i,j)] = 1
                if self.W[i][j] == 0:
                    if i > 0: t[flat(i-1,j)] = 1
                    if j > 0: t[flat(i,j-1)] = 1
                    if i < self.W.shape[0]-1: t[flat(i+1,j)] = 1
                    if j < self.W.shape[1]-1: t[flat(i,j+1)] = 1
                disc_dynamics.trans.append(t)
        disc_dynamics.num_prop = len(disc_dynamics.list_prop_symbol)
        disc_dynamics.num_regions = len(disc_dynamics.list_region)
        return disc_dynamics
        
    def spec(self, offset=(0, 0), controlled_dyn=True):
        """Return GRSpec instance describing this gridworld.

        The offset argument is motivated by the use-case of multiple
        agents whose moves are governed by separate "gridworlds" but
        who interact in a shared space; with an offset, we can make
        "sub-gridworlds" and enforce rules like mutual exclusion.

        Syntax is that of gr1c; in particular, "next" variables are
        primed. For example, x' refers to the variable x at the next
        time step.

        Variables are named according to prefix_R_C, where prefix is
        given (attribute of this GridWorld object), R is the row, and
        column the cell (0-indexed).

        For incorporating this gridworld into an existing
        specification (e.g., respecting external references to cell
        variable names), see the method L{GRSpec.importGridWorld}.

        @param offset: index offset to apply when generating the
                       specification; e.g., given prefix of "Y",
                       offset=(2,1) would cause the variable for the
                       cell at (0,3) to be named Y_2_4.

        @param controlled_dyn: whether to treat this gridworld as
                               describing controlled ("system") or
                               uncontrolled ("environment") variables.

        @rtype: L{GRSpec}
        """
        if self.W is None:
            raise ValueError("Gridworld does not exist.")
        row_low = 0
        row_high = self.W.shape[0]
        col_low = 0
        col_high = self.W.shape[1]
        spec_trans = []
        self.offset = offset
        # Safety, transitions
        for i in range(row_low, row_high):
            for j in range(col_low, col_high):
                if self.W[i][j] == 1:
                    continue  # Cannot start from an occupied cell.
                spec_trans.append(self[i,j]+" -> (")
                # Normal transitions:
                spec_trans[-1] += self[i,j]+"'"
                if i > row_low and self.W[i-1][j] == 0:
                    spec_trans[-1] += " | " + self[i-1,j]+"'"
                if j > col_low and self.W[i][j-1] == 0:
                    spec_trans[-1] += " | " + self[i,j-1]+"'"
                if i < row_high-1 and self.W[i+1][j] == 0:
                    spec_trans[-1] += " | " + self[i+1,j]+"'"
                if j < col_high-1 and self.W[i][j+1] == 0:
                    spec_trans[-1] += " | " + self[i,j+1]+"'"
                spec_trans[-1] += ")"

        # Safety, static
        for i in range(row_low, row_high):
            for j in range(col_low, col_high):
                if self.W[i][j] == 1:
                    spec_trans.append("!(" + self[i,j]+"'" + ")")

        # Safety, mutex
        pos_indices = [k for k in itertools.product(range(row_low, row_high), range(col_low, col_high))]
        disj = []
        for outer_ind in pos_indices:
            conj = []
            if outer_ind != (-1, -1) and self.W[outer_ind[0]][outer_ind[1]] == 1:
                continue
            if outer_ind == (-1, -1):
                conj.append(self.prefix+"_n_n'")
            else:
                conj.append(self[outer_ind[0], outer_ind[1]]+"'")
            for inner_ind in pos_indices:
                if ((inner_ind != (-1, -1) and self.W[inner_ind[0]][inner_ind[1]] == 1)
                    or outer_ind == inner_ind):
                    continue
                if inner_ind == (-1, -1):
                    conj.append("(!" + self.prefix+"_n_n')")
                else:
                    conj.append("(!" + self[inner_ind[0], inner_ind[1]]+"')")
            disj.append("(" + " & ".join(conj) + ")")
        spec_trans.append("\n| ".join(disj))

        sys_vars = []
        for i in range(row_low, row_high):
            for j in range(col_low, col_high):
                sys_vars.append(self[i,j])

        initspec = []
        for loc in self.init_list:
            mutex = [self[loc[0],loc[1]]]
            mutex.extend(["!"+ovar for ovar in sys_vars if ovar != self[loc]])
            initspec.append("(" + " & ".join(mutex) + ")")
        init_str = " | ".join(initspec)

        spec_goal = []
        for loc in self.goal_list:
            spec_goal.append(self[loc])
        
        #oldspec = self.spec_old(offset, controlled_dyn)
        #assert(spec_trans == oldspec.sys_safety)
        #assert(sys_vars == oldspec.sys_vars)
        #assert(init_str == oldspec.sys_init[0])
        #assert(spec_goal == oldspec.sys_prog)
        self.offset = (0, 0)
        if controlled_dyn:
            return GRSpec(sys_vars=sys_vars, sys_init=init_str,
                          sys_safety=spec_trans, sys_prog=spec_goal)
        else:
            return GRSpec(env_vars=sys_vars, env_init=init_str,
                          env_safety=spec_trans, env_prog=spec_goal)
                      
    def spec_old(self, offset=(0, 0), controlled_dyn=True):
        """Return GRSpec instance describing this gridworld.

        The offset argument is motivated by the use-case of multiple
        agents whose moves are governed by separate "gridworlds" but
        who interact in a shared space; with an offset, we can make
        "sub-gridworlds" and enforce rules like mutual exclusion.

        Syntax is that of gr1c; in particular, "next" variables are
        primed. For example, x' refers to the variable x at the next
        time step.

        Variables are named according to prefix_R_C, where prefix is
        given (attribute of this GridWorld object), R is the row, and
        column the cell (0-indexed).

        For incorporating this gridworld into an existing
        specification (e.g., respecting external references to cell
        variable names), see the method L{GRSpec.importGridWorld}.

        @param offset: index offset to apply when generating the
                       specification; e.g., given prefix of "Y",
                       offset=(2,1) would cause the variable for the
                       cell at (0,3) to be named Y_2_4.

        @param controlled_dyn: whether to treat this gridworld as
                               describing controlled ("system") or
                               uncontrolled ("environment") variables.

        @rtype: L{GRSpec}
        """
        if self.W is None:
            raise ValueError("Gridworld does not exist.")
        row_low = 0
        row_high = self.W.shape[0]-1
        col_low = 0
        col_high = self.W.shape[1]-1
        spec_trans = []
        # Safety, transitions
        for i in range(row_low, row_high+1):
            for j in range(col_low, col_high+1):
                if self.W[i][j] == 1:
                    continue  # Cannot start from an occupied cell.
                spec_trans.append(self.prefix+"_"+str(i+offset[0])+"_"+str(j+offset[1])+" -> (")
                # Normal transitions:
                spec_trans[-1] += self.prefix+"_"+str(i+offset[0])+"_"+str(j+offset[1])+"'"
                if i > row_low and self.W[i-1][j] == 0:
                    spec_trans[-1] += " | " + self.prefix+"_"+str(i-1+offset[0])+"_"+str(j+offset[1])+"'"
                if j > col_low and self.W[i][j-1] == 0:
                    spec_trans[-1] += " | " + self.prefix+"_"+str(i+offset[0])+"_"+str(j-1+offset[1])+"'"
                if i < row_high and self.W[i+1][j] == 0:
                    spec_trans[-1] += " | " + self.prefix+"_"+str(i+1+offset[0])+"_"+str(j+offset[1])+"'"
                if j < col_high and self.W[i][j+1] == 0:
                    spec_trans[-1] += " | " + self.prefix+"_"+str(i+offset[0])+"_"+str(j+1+offset[1])+"'"
                spec_trans[-1] += ")"

        # Safety, static
        for i in range(row_low, row_high+1):
            for j in range(col_low, col_high+1):
                if self.W[i][j] == 1:
                    spec_trans.append("!(" + self.prefix+"_"+str(i+offset[0])+"_"+str(j+offset[1])+"'" + ")")

        # Safety, mutex
        first_subformula = True
        spec_trans.append("")
        pos_indices = [k for k in itertools.product(range(row_low, row_high+1), range(col_low, col_high+1))]
        for outer_ind in pos_indices:
            if outer_ind != (-1, -1) and self.W[outer_ind[0]][outer_ind[1]] == 1:
                continue
            if not first_subformula:
                spec_trans[-1] += "\n| "
            if outer_ind == (-1, -1):
                spec_trans[-1] += "(" + self.prefix+"_n_n'"
            else:
                spec_trans[-1] += "(" + self.prefix+"_"+str(outer_ind[0]+offset[0])+"_"+str(outer_ind[1]+offset[1])+"'"
            for inner_ind in pos_indices:
                if ((inner_ind != (-1, -1) and self.W[inner_ind[0]][inner_ind[1]] == 1)
                    or outer_ind == inner_ind):
                    continue
                if inner_ind == (-1, -1):
                    spec_trans[-1] += " & (!" + self.prefix+"_n_n')"
                else:
                    spec_trans[-1] += " & (!" + self.prefix+"_"+str(inner_ind[0]+offset[0])+"_"+str(inner_ind[1]+offset[1])+"'" + ")"
            spec_trans[-1] += ")"
            first_subformula = False

        pvars = []
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                pvars.append(self.prefix+"_"+str(i+offset[0])+"_"+str(j+offset[1]))

        init_str = ""
        for loc in self.init_list:
            if len(init_str) > 0:
                init_str += " | "
            init_str += "(" + self.prefix+"_"+str(loc[0]+offset[0])+"_"+str(loc[1]+offset[1])
            init_str_mutex = " & ".join(["!"+ovar for ovar in pvars if ovar != self.prefix+"_"+str(loc[0]+offset[0])+"_"+str(loc[1]+offset[1])])
            if len(init_str_mutex) > 0:
                init_str += " & " + init_str_mutex
            init_str += ")"

        spec_goal = []
        for loc in self.goal_list:
            spec_goal.append(self.prefix+"_"+str(loc[0]+offset[0])+"_"+str(loc[1]+offset[1]))

        if controlled_dyn:
            return GRSpec(sys_vars=pvars, sys_init=init_str,
                          sys_safety=spec_trans, sys_prog=spec_goal)
        else:
            return GRSpec(env_vars=pvars, env_init=init_str,
                          env_safety=spec_trans, env_prog=spec_goal)


def random_world(size, wall_density=.2, num_init=1, num_goals=2, prefix="Y"):
    """Generate random gridworld of given size.

    While an instance of GridWorld is returned, other views of the
    result are possible; e.g., to obtain a description string, use
    L{GridWorld.dumps}.

    @param size: a pair, indicating number of rows and columns.
    @param wall_density: the ratio of walls to total number of cells.
    @param num_init: number of possible initial positions.
    @param num_goals: number of positions to be visited infinitely often.
    @param prefix: string to be used as prefix for naming gridworld
                   cell variables.

    @rtype: L{GridWorld}
    """
    num_cells = size[0]*size[1]
    goal_list = []
    init_list = []
    W = np.zeros(num_cells, dtype=np.int32)
    num_blocks = int(np.round(wall_density*num_cells))
    try:
        for i in range(num_blocks):
            avail_inds = np.array(range(num_cells))[W==0]
            W[avail_inds[np.random.randint(low=0, high=len(avail_inds))]] = 1
        for i in range(num_goals):
            avail_inds = np.array(range(num_cells))[W==0]
            avail_inds = [k for k in avail_inds if k not in goal_list]
            goal_list.append(avail_inds[np.random.randint(low=0, high=len(avail_inds))])
        for i in range(num_init):
            avail_inds = np.array(range(num_cells))[W==0]
            avail_inds = [k for k in avail_inds if k not in goal_list and k not in init_list]
            init_list.append(avail_inds[np.random.randint(low=0, high=len(avail_inds))])
    except ValueError:
        # We've run out of available indices, so cannot produce a world
        raise ValueError("World too small for number of features")
    W = W.reshape(size)
    goal_list = [(k/size[1], k%size[1]) for k in goal_list]
    init_list = [(k/size[1], k%size[1]) for k in init_list]
    gw = GridWorld(prefix=prefix)
    gw.W = W
    gw.goal_list = goal_list
    gw.init_list = init_list
    return gw
    
def narrow_passage(size, passage_width=1, num_init=1, num_goals=2, prefix="Y"):
    """Generate a narrow-passage world: this is a world containing 
    two zones (initial, final) with a tube connecting them.
    
    @param size: a pair, indicating number of rows and columns.
    @param tube_width: the width of the connecting tube in cells.
    @param num_init: number of possible initial positions.
    @param num_goals: number of positions to be visited infinitely often.
    @param prefix: string to be used as prefix for naming gridworld
                   cell variables.
                   
    @rtype: L{GridWorld}"""
                   
    (w, h) = size
    if w < 3 or h < 3:
        raise ValueError("Gridworld too small: minimum dimension 3")
    Z = unoccupied(size, prefix)
    izone = int(max(1, 0.3*size[1])) # boundary of left zone, 20% of width
    gzone = size[1] - int(max(1, 0.3*size[1])) # boundary of right zone
    if izone * size[0] < num_init or gzone * size[0] < num_goals:
        raise ValueError("Too many initials/goals for grid size")
    ptop = np.random.randint(0, size[1]-passage_width)
    passage = range(ptop, ptop+passage_width)
    for y in range(0, size[0]):
        if y not in passage:
            for x in range(izone, gzone):
                Z.W[y][x] = 1
    avail_cells = [(y,x) for y in range(size[0]) for x in range(izone)]
    Z.init_list = random.sample(avail_cells, num_init)
    avail_cells = [(y,x) for y in range(size[0]) for x in range(gzone, size[1])]
    Z.goal_list = random.sample(avail_cells, num_goals)
    return Z

def unoccupied(size, prefix="Y"):
    """Generate entirely unoccupied gridworld of given size.
    
    @param size: a pair, indicating number of rows and columns.
    @param prefix: String to be used as prefix for naming gridworld
                   cell variables.
    @rtype: L{GridWorld}
    """
    if len(size) < 2:
        raise TypeError("invalid gridworld size.")
    return GridWorld(str(size[0])+" "+str(size[1]), prefix="Y")

def add_trolls(Y, troll_list, prefix="X"):
    """Create GR(1) specification with troll-like obstacles.

    Trolls are introduced into the specification with names derived
    from the given prefix and a number (matching the order in troll_list).

    @type Y: L{GridWorld}
    @param Y: The controlled gridworld, describing in particular
              static obstacles that must be respected by the trolls.

    @param troll_list: List of pairs of center position, to which the
                       troll must always eventually return, and radius
                       defining the extent of the trollspace.  The
                       radius is measured using infinity-norm.
    
    @rtype: (L{GRSpec}, list)

    @return: Returns (spec, moves_N) where spec is the specification
             incorporating all of the trolls, and moves_N is a list of
             lists of states (where "state" is given as a dictionary
             with keys of variable names), where the length of moves_N
             is equal to the number of trolls, and each element of
             moves_N is a list of possible states of that the
             corresponding troll (dynamic obstacle).
    """
    X = []
    X_ID = -1
    moves_N = []
    (num_rows, num_cols) = Y.size()
    for (center, radius) in troll_list:
        if center[0] >= num_rows or center[0] < 0 or center[1] >= num_cols or center[1] < 0:
            raise ValueError("troll center is outside of gridworld")
        t_offset = (max(0, center[0]-radius), max(0, center[1]-radius))
        t_size = [center[0]-t_offset[0]+radius+1, center[1]-t_offset[1]+radius+1]
        if t_offset[0]+t_size[0] >= num_rows:
            t_size[0] = num_rows-t_offset[0]
        if t_offset[1]+t_size[1] >= num_cols:
            t_size[1] = num_cols-t_offset[1]
        t_size = (t_size[0], t_size[1])
        X_ID += 1
        X.append((t_offset, Y.dumpsubworld(t_size, offset=t_offset, prefix=prefix+"_"+str(X_ID))))
        X[-1][1].goal_list = [(center[0]-t_offset[0], center[1]-t_offset[1])]
        X[-1][1].init_list = [(center[0]-t_offset[0], center[1]-t_offset[1])]
        moves_N.append([])
        for i in range(t_size[0]):
            for j in range(t_size[1]):
                moves_N[-1].append(X[-1][1].state((i,j), offset=t_offset))

    spec = GRSpec()
    spec.importGridWorld(Y, controlled_dyn=True)
    for Xi in X:
        spec.importGridWorld(Xi[1], offset=Xi[0], controlled_dyn=False)

    # Mutual exclusion
    for i in range(Y.size()[0]):
        for j in range(Y.size()[1]):
            for Xi in X:
                if i >= Xi[0][0] and i < Xi[0][0]+Xi[1].size()[0] and j >= Xi[0][1] and j < Xi[0][1]+Xi[1].size()[1]:
                    spec.sys_safety.append("!("+Y[i,j]+"' & "+Xi[1].prefix+"_"+str(i)+"_"+str(j)+"')")

    return (spec, moves_N)


def unoccupied(size, prefix="Y"):
    """Generate entirely unoccupied gridworld of given size.
    
    @param size: a pair, indicating number of rows and columns.
    @param prefix: String to be used as prefix for naming gridworld
                   cell variables.
    @rtype: L{GridWorld}
    """
    if len(size) < 2:
        raise TypeError("invalid gridworld size.")
    return GridWorld(str(size[0])+" "+str(size[1]), prefix="Y")


def extract_coord(var_name):
    """Assuming prefix_R_C format, return (prefix,row,column) tuple.

    prefix is of type string, row and column are integers.

    The "nowhere" coordinate has form prefix_n_n. To indicate this,
    (-1, -1) is returned as the row, column position.

    If error, return None or throw exception.
    """
    if not isinstance(var_name, str):
        raise TypeError("extract_coord: invalid argument type; must be string.")
    name_frags = var_name.split("_")
    if len(name_frags) < 3:
        return None
    try:
        if name_frags[-1] == "n" and name_frags[-2] == "n":
            # Special "nowhere" case
            return ("_".join(name_frags[:-2]), -1, -1)
        col = int(name_frags[-1])
        row = int(name_frags[-2])
    except ValueError:
        return None
    return ("_".join(name_frags[:-2]), row, col)

def prefix_filt(d, prefix):
    """Return all items in dictionary d with key with given prefix."""
    match_list = []
    for k in d.keys():
        if isinstance(k, str):
            if k.startswith(prefix):
                match_list.append(k)
    return dict([(k, d[k]) for k in match_list])
    
def extractPath(aut, prefix=None):
    """Extract a path from a gridworld automaton"""
    s = aut.getAutState(0)
    last = None
    path = []
    visited = [0]
    while 1:
        updated = False
        for p in s.state:
            if (not prefix or p.startswith(prefix)) and s.state[p]:
                try:
                    c = extract_coord(p)
                    if c:
                        path.append(c[1:])
                        last = c[1:]
                        updated = True
                except:
                    pass
        if not updated:
            # Robot has not moved, even out path lengths
            path.append(last)
        # next state
        if len(s.transition) > 0:
            if s.transition[0] in visited:
                # loop detected
                break
            visited.append(s.transition[0])
            s = aut.getAutState(s.transition[0])
        else:
            # dead-end, return
            break
    first = [ x for x in path if x ][0]
    for i in range(len(path)):
        if path[i] is None:
            path[i] = first
        else:
            break
    return path
