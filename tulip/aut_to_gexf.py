#! /usr/bin/env/python

'''
Convert an AUT file (from JTLV) to a GEXF (in Gephi) file.

Accepts a *.aut filename and, optionally, a destination filename. If no
destination is given, default is to use input filename with new ending ".gexf"

Also note that command-line arguments are handled manually, since they
are so simple...

Originally written by Scott Livingston for GraphViz DOT files.f6f

Modified by Yuchen Lin for Gephi GEXF files.
'''

import sys
from tulip.automaton import Automaton
from tulip.errorprint import printWarning, printError

class Automaton_Gephi(Automaton):
    '''
    Automaton_Gephi extends Automaton by adding functions to write the
    current automaton as a Gephi 'gexf' graph file.
    '''

    def autVar2GexfAttr(self, var_name, var_type):
        '''
        Given the string name of a variable (var_name) and a Python type
        (var_type), converts the Python type to a gexf-accepted string and
        encodes them in an gexf-readable attribute.
        '''
        # First create a dictionary that translates Python data type names
        # to gexf data type names.
        type_dict = {str: 'string', int: 'integer',
                     float: 'float', bool: 'boolean'}

        # Generate a line of XML for the attribute.
        attribute = '\n        <attribute id="' + var_name \
               + '" type="' + type_dict[var_type] + '" />'
        return attribute
        
    
    def autState2GexfNode(self, state):
        '''
        Given an AutomatonState 'state', encodes the fields of that state in
        a gexf-readable node.

        The variables in the 'state.state' dictionary are stored as attributes.
        '''
        # Generate a line of XML for the node.
        node = '\n        <node id="' + str(state.id) \
               + '" label="' + str(state.state) + '">'
        node += '\n          <attvalues>'

        # Iterate through attributes.
        for (k, v) in state.state.items():
            node += '\n            <attvalue for="' + k \
                    + '" value="' + str(v) + '" />'

        # Close attributes and node.
        node += '\n          </attvalues>'
        node += '\n        </node>'
        return node
        
    
    def autTrans2GexfEdge(self, source, target_id, edge_id):
        '''
        Given an AutomatonState 'source', an AutomatonState ID 'target_id',
        and an integer that represents the desired edge number, encodes the
        transition between them, if it exists, in a gexf-readable edge.

        The variables in the TARGET's 'state.state' dictionary are stored as
        attributes.
        '''
        # Locate target node.
        for state in self.states:
            if target_id == state.id:
                target = state
                # Since target AutomatonState already found, leave for loop.
                break
        
        # Generate a line of XML for the edge.
        edge = '\n        <edge id="' + str(edge_id) \
                            + '" source="' + str(source.id) \
                            + '" target="' + str(target.id) \
                            + '" label="' + str(target.state) + '">'
        edge += '\n          <attvalues>'

        # Iterate through attributes.
        for (k, v) in target.state.items():
            edge += '\n            <attvalue for="' + k \
                    + '" value="' + str(v) + '" />'
        
        # Close attributes and node.
        edge += '\n          </attvalues>'
        edge += '\n        </edge>'
        return edge

    
    def writeGexfFile(self, destfile):
        '''
        Writes the automaton to a Gephi 'gexf' file named 'destfile'
        with system states on the nodes and environment states on the edges.
        '''
        
        try:
            f = open(destfile, "w")
        except:
            printWarning("Failed to open " + destfile + " for writing.", obj=self)
            return False
        
        # 'gexf_header', 'gexf_divider_0', 'gexf_divider_1',
        # 'gexf_divider_2', and 'gexf_footer are all used to format the XML
        # of the final gexf file.
        
        gexf_header = '''<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://www.gexf.net/1.2draft"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://www.gexf.net/1.2draft
                          http://www.gexf.net/1.2draft/gexf.xsd"
      version="1.2">
    <meta lastmodifieddate="2011-06-20">
        <creator>Yuchen Lin</creator>
        <description>Automaton in gexf.</description>
    </meta>
    <graph defaultedgetype="directed">
      <attributes class="node">'''

        gexf_divider_0 = '''
      </attributes>
      <attributes class="edge">'''
        
        gexf_divider_1 = '''
      </attributes>
      <nodes>'''
        
        gexf_divider_2 = '''
      </nodes>
      <edges>'''
        
        gexf_footer = '''
      </edges>
    </graph>
</gexf>'''
        
        try:
            f.write(gexf_header)

            # Build gexf node attributes (used to specify the types of data
            # that can be stored) from the 'state' dictionary of
            # AutomatonState states.
            node_attr_list = []
            for state in self.states:
                for (k, v) in state.state.items():
                    if k not in node_attr_list:
                        f.write(self.autVar2GexfAttr(k, type(v)))
                        node_attr_list.append(k)
            
            f.write(gexf_divider_0)

            # Build gexf edge attributes (used to specify the types of data
            # that can be stored) from the 'state' dictionary of
            # AutomatonState states.
            edge_attr_list = []
            for state in self.states:
                for (k, v) in state.state.items():
                    if k not in edge_attr_list:
                        f.write(self.autVar2GexfAttr(k, type(v)))
                        edge_attr_list.append(k)
            
            f.write(gexf_divider_1)
            
            # Build gexf nodes from AutomatonState states.
            for state in self.states:
                f.write(self.autState2GexfNode(state))

            f.write(gexf_divider_2)
            
            # Build gexf edges from AutomatonState transitions.
            edge_id = 0
            for state in self.states:
                for trans in state.transition:
                    f.write(self.autTrans2GexfEdge(state, trans, edge_id))
                    edge_id += 1
            
            f.write(gexf_footer)
        except:
            f.close()
            printWarning("Error occurred while generating GEXF code for automaton.", obj=self)
            return False

        f.close()
        return True



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
    
    aut = Automaton_Gephi(sys.argv[1])
    if not aut.writeGexfFile(destfile):
        print "Failed to create GEXF file, " + destfile
        exit(-1)
