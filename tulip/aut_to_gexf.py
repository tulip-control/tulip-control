#! /usr/bin/env python
#
# Copyright (c) 2011 by California Institute of Technology
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
Convert an AUT file (from JTLV) to a GEXF (in Gephi) file.

Accepts a *.aut filename and, optionally, a destination filename. If no
destination is given, default is to use input filename with new ending ".gexf"

Also note that command-line arguments are handled manually, since they
are so simple...

Originally written by Scott Livingston for GraphViz DOT files.

Modified by Yuchen Lin for Gephi GEXF files.

"""

import sys
from subprocess import call

from tulip.automaton import AutomatonState, Automaton
from tulip.errorprint import printWarning, printError

class Automaton_Gephi(Automaton):
    """
    Automaton_Gephi extends Automaton by adding functions to write the
    current automaton as a Gephi 'gexf' graph file.

    Note that the graph visualization software Gephi should be installed
    for this class to be useful.
    """

    def tagGexfAttr(self, var_name, var_type, idt_lvl=3):
        """
        Encode the variable specified in a gexf-readable attribute.

        Arguments:
        var_name -- the string name of the desired variable
        var_type -- Python type of the  variable
        idt_lvl -- level of indentation of the attribute. Default is 3,
            as gexf attributes are usually three levels in.

        Return:
        String representing a gexf attribute.
        """

        if not (isinstance(var_name, str) and isinstance(var_type, type) and
                isinstance(idt_lvl, int)):
            raise TypeError("Invalid arguments to tagGexfAttr")
        
        nl = "\n"  # newline
        idt = "  "  # indentation
        
        # First create a dictionary that translates Python data type names
        # to gexf data type names.
        type_dict = {str: 'string', int: 'integer',
                     float: 'float', bool: 'boolean'}

        # Generate a line of XML for the attribute.
        attribute = idt * idt_lvl + '<attribute id="' + var_name + \
                    '" type="' + type_dict[var_type] + '" />' + nl
        
        return attribute
        
    
    def tagGexfNode(self, state, label, idt_lvl=3):
        """
        Encode the state specified in a gexf-readable node. The variables
        in the specified state are stored as attribute values.

        Arguments:
        state -- the AutomatonState object to be encoded
        label -- a string name for the node.
        idt_lvl -- level of indentation of the node. Default is 3,
            as gexf nodes are usually three levels in.

        Return:
        String representing a gexf node.
        """
        
        if not (isinstance(state, AutomatonState) and
                isinstance(label, str) and isinstance(idt_lvl, int)):
            raise TypeError("Invalid arguments to tagGexfNode")
        
        nl = "\n"  # newline
        idt = "  "  # indentation
        
        # Generate a line of XML for the node.
        node = idt * idt_lvl + '<node id="' + str(state.id) + \
               '" label="' + label + '">' + nl
        idt_lvl += 1
        node += idt * idt_lvl + '<attvalues>' + nl

        # Iterate through attributes.
        idt_lvl += 1
        for (k, v) in state.state.items():
            node += idt * idt_lvl + '<attvalue for="' + k + \
                    '" value="' + str(v) + '" />' + nl
        idt_lvl -= 1

        # Close attributes and node.
        node += idt * idt_lvl + '</attvalues>' + nl
        idt_lvl -= 1
        node += idt * idt_lvl + '</node>' + nl
        
        return node
        
    
    def tagGexfEdge(self, source, target, edge_id, label, idt_lvl=3):
        """
        Encode the transition specified in a gexf-readable edge. The variables
        in the target's state are stored as attribute values.

        Arguments:
        source -- the AutomatonState for the 'tail' of the edge.
        target -- the AutomatonState for the 'head' of the edge.
        edge_id -- the ID number of the desired edge.
        label -- a string name for the edge.
        idt_lvl -- level of indentation of the edge. Default is 3,
            as gexf edges are usually three levels in.

        Return:
        String representing a gexf edge.
        """
        
        if not (isinstance(source, AutomatonState) and
                isinstance(target, AutomatonState) and
                isinstance(edge_id, int) and isinstance(label, str) and
                isinstance(idt_lvl, int)):
            raise TypeError("Invalid arguments to tagGexfEdge")
        
        nl = "\n"  # newline
        idt = "  "  # indentation
        
        # Generate a line of XML for the edge.
        edge = idt * idt_lvl + '<edge id="' + str(edge_id) + \
               '" source="' + str(source.id) + \
               '" target="' + str(target.id) + \
               '" label="' + label + '">' + nl
        idt_lvl += 1
        edge += idt * idt_lvl + '<attvalues>' + nl

        # Iterate through attributes.
        idt_lvl += 1
        for (k, v) in target.state.items():
            edge += idt * idt_lvl + '<attvalue for="' + k + \
                    '" value="' + str(v) + '" />' + nl
        idt_lvl -= 1
        
        # Close attributes and node.
        edge += idt * idt_lvl + '</attvalues>' + nl
        idt_lvl -= 1
        edge += idt * idt_lvl + '</edge>' + nl
        
        return edge

    
    def writeGexfFile(self, destfile):
        """
        Writes the automaton to a Gephi 'gexf' file. Nodes represent system
        states and edges represent transitions between nodes.

        Arguments:
        destfile -- string path to the desired destination file

        Return:
        (nothing)
        """

        if not isinstance(destfile, str):
            raise TypeError("Argument to writeGexfFile must be a string.")
        
        try:
            f = open(destfile, "w")
        except:
            printWarning("Failed to open " + destfile + " for writing.", obj=self)
            return False
        
        
        nl = "\n"  # newline
        idt = "  "  # indentation
        idt_lvl = 0 # indentation level
        output = '' # string to be written to file
        
        # Open xml, gexf, and node attributes tags.
        output += idt * idt_lvl + '<?xml version="1.0" encoding="UTF-8"?>' + nl
        output += idt * idt_lvl + '<gexf version="1.2">' + nl
        idt_lvl += 1
        output += idt * idt_lvl + '<graph defaultedgetype="directed">' + nl
        idt_lvl += 1
        output += idt * idt_lvl + '<attributes class="node">' + nl

        # Build gexf node attributes (used to specify the types of data
        # that can be stored) from the 'state' dictionary of
        # AutomatonState states.
        idt_lvl += 1
        node_attr_list = []
        for state in self.states:
            for (k, v) in state.state.items():
                if k not in node_attr_list:
                    output += self.tagGexfAttr(k, type(v), idt_lvl=idt_lvl)
                    node_attr_list.append(k)
        idt_lvl -= 1
        
        # Close node attributes tag and open edge attributes tag.
        output += idt * idt_lvl + '</attributes>' + nl
        output += idt * idt_lvl + '<attributes class="edge">' + nl

        # Build gexf edge attributes (used to specify the types of data
        # that can be stored) from the 'state' dictionary of
        # AutomatonState states.
        idt_lvl += 1
        edge_attr_list = []
        for state in self.states:
            for (k, v) in state.state.items():
                if k not in edge_attr_list:
                    output += self.tagGexfAttr(k, type(v), idt_lvl=idt_lvl)
                    edge_attr_list.append(k)
        idt_lvl -= 1
        
        # Close edge attributes tag and open nodes tag.
        output += idt * idt_lvl + '</attributes>' + nl
        output += idt * idt_lvl + '<nodes>' + nl
        
        # Build gexf nodes from AutomatonState states.
        idt_lvl += 1
        for state in self.states:
            output += self.tagGexfNode(state, str(state.state),
                                       idt_lvl=idt_lvl)
        idt_lvl -= 1
        
        # Close nodes tag and open edges tag.
        output += idt * idt_lvl + '</nodes>' + nl
        output += idt * idt_lvl + '<edges>' + nl
        
        # Build gexf edges from AutomatonState transitions.
        idt_lvl += 1
        edge_id = 0  # Each edge is numbered, starting from 0.
        for state in self.states:
            for trans in state.transition:
                # Locate target node.
                for aut_state in self.states:
                    if trans == aut_state.id:
                        target = aut_state
                        # target has been found, leave for loop.
                        break
                output += self.tagGexfEdge(state, target, edge_id,
                                           str(target.state), idt_lvl=idt_lvl)
                edge_id += 1
        idt_lvl -= 1

        # Close edges, graph, and gexf tags.
        output += idt * idt_lvl + '</edges>' + nl
        idt_lvl -= 1
        output += idt * idt_lvl + '</graph>' + nl
        idt_lvl -= 1
        output += idt * idt_lvl + '</gexf>' + nl

        assert idt_lvl == 0


        # Write output to file.
        try:
            f.write(output)
            f.close()
            return True
        except:
            f.close()
            printWarning("Error occurred while generating GEXF code for automaton.", obj=self)
            return False



if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print "Usage: aut_to_gexf.py INPUT.aut [OUTPUT.gexf]"
        exit(1)

    if len(sys.argv) == 3:
        destfile = sys.argv[2]
    else:
        if len(sys.argv[1]) < 4:  # Handle weirdly short names
            destfile = sys.argv[1] + ".gexf"
        else:
            destfile = sys.argv[1][:-4] + ".gexf"
    
    testAutState0 = AutomatonState(id=0, state={'a':0, 'b':1}, transition=[1])
    testAutState1 = AutomatonState(id=1, state={'a':2, 'b':3}, transition=[0])
    testAut = Automaton_Gephi([testAutState0, testAutState1])    
    print "Testing tagGexfAttr..."
    assert testAut.tagGexfAttr('a', int, idt_lvl=3) == \
           '      <attribute id="a" type="integer" />\n'
    print "Testing tagGexfNode..."
    assert testAut.tagGexfNode(testAutState0, 'foo', idt_lvl=3) == \
           '      <node id="0" label="foo">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="0" />\n' + \
           '          <attvalue for="b" value="1" />\n' + \
           '        </attvalues>\n' + \
           '      </node>\n'
    print "Testing tagGexfEdge..."
    assert testAut.tagGexfEdge(testAutState0, testAutState1,
                               0, 'bar', idt_lvl=3) == \
           '      <edge id="0" source="0" target="1" label="bar">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="2" />\n' + \
           '          <attvalue for="b" value="3" />\n' + \
           '        </attvalues>\n' + \
           '      </edge>\n'
    print "Testing writeGexfFile..."
    # Writes to the output file specified. This file will be replaced later.
    assert str(testAut.writeGexfFile(destfile))
    print "Tests passed."
    
    aut = Automaton_Gephi(sys.argv[1])
    print "Generating GEXF file."
    if not aut.writeGexfFile(destfile):
        print "Failed to create GEXF file, " + destfile
        exit(-1)
##    else:
##        try:
##            print "Opening GEXF file in Gephi."
##            call(["gephi", destfile])
##        except:
##            print "Failed to open " + destfile + " in Gephi."
