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
import numpy as np
from spec import GRSpec


class GridWorld:
    def __init__(self, gw_desc=None, prefix="Y"):
        """Load gridworld described in given string, or make empty instance.

        @param gw_desc: String containing a gridworld description, or
                        None to create an empty instance.
        @param prefix: String to be used as prefix for naming
                       gridworld cell variables
        """
        if gw_desc is not None:
            self.loads(gw_desc)
        else:
            self.W = None
            self.init_list = []
            self.goal_list = []
        self.prefix = prefix

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
        """Return variable name corresponding to this cell."""
        if self.W is None:
            raise ValueError("Gridworld is empty; no names available.")
        if len(key) != len(self.W.shape):
            raise ValueError("malformed gridworld key.")
        if key[0] < 0 or key[1] < 0 or key[0] >= self.W.shape[0] or key[1] >= self.W.shape[1]:
            raise ValueError("gridworld key is out of bounds.")
        return str(self.prefix)+"_"+"_".join([str(i) for i in key])


    def isEmpty(self, coord):
        """Is cell at coord empty?
        
        @param coord: (row, column) pair.
        """
        if self.W is None:
            raise ValueError("Gridworld is empty; no cells exist.")
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
        
    def pretty(self, show_grid=False):
        """Return pretty-for-printing string.

        @param show_grid: If True, then grid the pretty world and show
                          row and column labels along the outer edges.
        """
        # See comments in code for the method loads regarding values in W
        if self.W is None:
            return ""
        
        # LEGEND:
        #  * - wall (as used in original world matrix definition);
        #  G - goal location;
        #  I - possible initial location.
        if show_grid:
            out_str = "  " + "".join([str(k).rjust(2) for k in range(self.W.shape[1])]) + "\n"
        else:
            out_str = "-"*(self.W.shape[1]+2) + "\n"
        for i in range(self.W.shape[0]):
            if show_grid:
                out_str += "  " + "-"*(self.W.shape[1]*2+1) + "\n"
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
                        out_str += "G"
                    else:
                        out_str += " "
                elif self.W[i][j] == 1:
                    out_str += "*"
                else:
                    raise ValueError("Unrecognized internal world W encoding.")
            out_str += "|\n"
        if show_grid:
            out_str += "  " + "-"*(self.W.shape[1]*2+1) + "\n"
        else:
            out_str += "-"*(self.W.shape[1]+2) + "\n"
        return out_str
    

    def size(self):
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

        The second non-blank line is used to construct the first row
        of the gridworld, the third non-blank line to construct the
        second row, and so on. A row definition is
        whitespace-sensitive up to the number of columns (any
        characters beyond the column count are ignored, so in
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
            if len(line.lstrip()) > 0 and line.lstrip()[0] == "#":
                continue  # Ignore comment lines
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
                if len(line.strip()) == 0:
                    continue
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

    def dumps(self):
        """Dump gridworld description string."""
        if self.W is None:
            raise ValueError("Gridworld does not exist.")
        out_str = " ".join([str(i) for i in self.W.shape])+"\n"
        for i in range(self.W.shape[0]):
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

    def spec(self):
        """Return GRSpec instance describing this gridworld.

        Syntax is that of gr1c; in particular, "next" variables are
        primed. For example, x' refers to the variable x at the next
        time step.

        Variables are named according to prefix_R_C, where prefix is
        given (attribute of this GridWorld object), R is the row, and
        column the cell (0-indexed).

        For incorporating this gridworld into an existing
        specification (e.g., respecting external references to cell
        variable names), see the method importGridWorld of class GRSpec.
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
                spec_trans.append(self.prefix+"_"+str(i)+"_"+str(j)+" -> (")
                # Normal transitions:
                spec_trans[-1] += self.prefix+"_"+str(i)+"_"+str(j)+"'"
                if i > row_low and self.W[i-1][j] == 0:
                    spec_trans[-1] += " | " + self.prefix+"_"+str(i-1)+"_"+str(j)+"'"
                if j > col_low and self.W[i][j-1] == 0:
                    spec_trans[-1] += " | " + self.prefix+"_"+str(i)+"_"+str(j-1)+"'"
                if i < row_high and self.W[i+1][j] == 0:
                    spec_trans[-1] += " | " + self.prefix+"_"+str(i+1)+"_"+str(j)+"'"
                if j < col_high and self.W[i][j+1] == 0:
                    spec_trans[-1] += " | " + self.prefix+"_"+str(i)+"_"+str(j+1)+"'"
                spec_trans[-1] += ")"

        # Safety, static
        for i in range(row_low, row_high+1):
            for j in range(col_low, col_high+1):
                if self.W[i][j] == 1:
                    spec_trans.append("!(" + self.prefix+"_"+str(i)+"_"+str(j)+"'" + ")")

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
                spec_trans[-1] += "(" + self.prefix+"_"+str(outer_ind[0])+"_"+str(outer_ind[1])+"'"
            for inner_ind in pos_indices:
                if ((inner_ind != (-1, -1) and self.W[inner_ind[0]][inner_ind[1]] == 1)
                    or outer_ind == inner_ind):
                    continue
                if inner_ind == (-1, -1):
                    spec_trans[-1] += " & (!" + self.prefix+"_n_n')"
                else:
                    spec_trans[-1] += " & (!" + self.prefix+"_"+str(inner_ind[0])+"_"+str(inner_ind[1])+"'" + ")"
            spec_trans[-1] += ")"
            first_subformula = False

        sys_vars = []
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                sys_vars.append(self.prefix+"_"+str(i)+"_"+str(j))

        init_str = ""
        for loc in self.init_list:
            if len(init_str) > 0:
                init_str += " | "
            init_str += "(" + self.prefix+"_"+str(loc[0])+"_"+str(loc[1])
            init_str_mutex = " & ".join(["!"+ovar for ovar in sys_vars if ovar != self.prefix+"_"+str(loc[0])+"_"+str(loc[1])])
            if len(init_str_mutex) > 0:
                init_str += " & " + init_str_mutex
            init_str += ")"

        spec_goal = []
        for loc in self.goal_list:
            spec_goal.append(self.prefix+"_"+str(loc[0])+"_"+str(loc[1]))

        return GRSpec(sys_vars=sys_vars, sys_init=init_str,
                      sys_safety=spec_trans, sys_prog=spec_goal)


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
    W = np.zeros(num_cells, dtype=np.int32)
    num_blocks = int(np.round(wall_density*num_cells))
    for i in range(num_blocks):
        avail_inds = np.array(range(num_cells))[W==0]
        W[avail_inds[np.random.randint(low=0, high=len(avail_inds))]] = 1
    for i in range(num_goals):
        avail_inds = np.array(range(num_cells))[W==0]
        avail_inds = [k for k in avail_inds if k not in goal_list]
        goal_list.append(avail_inds[np.random.randint(low=0, high=len(avail_inds))])
    avail_inds = np.array(range(num_cells))[W==0]
    avail_inds = [k for k in avail_inds if (k not in goal_list)]
    init_list = [avail_inds[np.random.randint(low=0, high=len(avail_inds))]]
    W = W.reshape(size)
    goal_list = [(k/size[1], k%size[1]) for k in goal_list]
    init_list = [(k/size[1], k%size[1]) for k in init_list]
    gw = GridWorld(prefix=prefix)
    gw.W = W
    gw.goal_list = goal_list
    gw.init_list = init_list
    return gw


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
