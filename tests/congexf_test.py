#!/usr/bin/env python
"""
Test code moved here verbatim from bottom of tulip/congexf.py
Original code by Yuchen Lin, 2011.

SCL; 31 Dec 2011.
"""

from tulip.congexf import * 
from tulip.automaton import AutomatonState, Automaton


# This test should be broken into units.
def congexf_test():
    testAutState0 = AutomatonState(id=0, state={'a':0, 'b':1}, transition=[1])
    testAutState1 = AutomatonState(id=1, state={'a':2, 'b':3}, transition=[0])
    testAut = Automaton([testAutState0, testAutState1])
    print "Testing tagGexfAttr..."
    assert tagGexfAttr('a', int, 3) == \
           '      <attribute id="a" type="integer" />\n'   
    print "Testing tagGexfAttvalue..."
    assert tagGexfAttvalue('a', 0, 5) == \
           '          <attvalue for="a" value="0" />\n'
    print "Testing tagGexfNode..."
    assert tagGexfNode(0, testAutState0, 'foo', 3) == \
           '      <node id="0.0" label="foo" pid="0">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="0" />\n' + \
           '          <attvalue for="b" value="1" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </node>\n'
    
    print "Testing tagGexfEdge..."
    assert tagGexfEdge(0, testAutState0, 0, testAutState1, 'bar', 3) == \
           '      <edge id="0.0-0.1" source="0.0" target="0.1" label="bar">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="2" />\n' + \
           '          <attvalue for="b" value="3" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </edge>\n'
    print "Testing dumpGexf..."
    assert dumpGexf(testAut) == \
           '<?xml version="1.0" encoding="UTF-8"?>\n' + \
           '<gexf version="1.2">\n' + \
           '  <graph defaultedgetype="directed">\n' + \
           '    <attributes class="node">\n' + \
           '      <attribute id="a" type="integer" />\n' + \
           '      <attribute id="b" type="integer" />\n' + \
           '      <attribute id="is_active" type="integer" />\n' + \
           '    </attributes>\n' + \
           '    <attributes class="edge">\n' + \
           '      <attribute id="a" type="integer" />\n' + \
           '      <attribute id="b" type="integer" />\n' + \
           '      <attribute id="is_active" type="integer" />\n' + \
           '    </attributes>\n' + \
           '    <nodes>\n' + \
           '      <node id="0" label="W0: [\'a\', \'b\']" />\n' + \
           '      <node id="0.0" label="[0, 1]" pid="0">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="0" />\n' + \
           '          <attvalue for="b" value="1" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </node>\n' + \
           '      <node id="0.1" label="[2, 3]" pid="0">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="2" />\n' + \
           '          <attvalue for="b" value="3" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </node>\n' + \
           '    </nodes>\n' + \
           '    <edges>\n' + \
           '      <edge id="0.0-0.1" source="0.0" target="0.1" label="[2, 3]">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="2" />\n' + \
           '          <attvalue for="b" value="3" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </edge>\n' + \
           '      <edge id="0.1-0.0" source="0.1" target="0.0" label="[0, 1]">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="0" />\n' + \
           '          <attvalue for="b" value="1" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </edge>\n' + \
           '    </edges>\n' + \
           '  </graph>\n' + \
           '</gexf>\n'
    print "Testing changeGexfAttvalue..."
    assert changeGexfAttvalue(dumpGexf(testAut), "a", 5,
                              node_id="0.0", edge_id="0.1-0.0") == \
           '<?xml version="1.0" encoding="UTF-8"?>\n' + \
           '<gexf version="1.2">\n' + \
           '  <graph defaultedgetype="directed">\n' + \
           '    <attributes class="node">\n' + \
           '      <attribute id="a" type="integer" />\n' + \
           '      <attribute id="b" type="integer" />\n' + \
           '      <attribute id="is_active" type="integer" />\n' + \
           '    </attributes>\n' + \
           '    <attributes class="edge">\n' + \
           '      <attribute id="a" type="integer" />\n' + \
           '      <attribute id="b" type="integer" />\n' + \
           '      <attribute id="is_active" type="integer" />\n' + \
           '    </attributes>\n' + \
           '    <nodes>\n' + \
           '      <node id="0" label="W0: [\'a\', \'b\']" />\n' + \
           '      <node id="0.0" label="[0, 1]" pid="0">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="5" />\n' + \
           '          <attvalue for="b" value="1" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </node>\n' + \
           '      <node id="0.1" label="[2, 3]" pid="0">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="2" />\n' + \
           '          <attvalue for="b" value="3" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </node>\n' + \
           '    </nodes>\n' + \
           '    <edges>\n' + \
           '      <edge id="0.0-0.1" source="0.0" target="0.1" label="[2, 3]">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="2" />\n' + \
           '          <attvalue for="b" value="3" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </edge>\n' + \
           '      <edge id="0.1-0.0" source="0.1" target="0.0" label="[0, 1]">\n' + \
           '        <attvalues>\n' + \
           '          <attvalue for="a" value="5" />\n' + \
           '          <attvalue for="b" value="1" />\n' + \
           '          <attvalue for="is_active" value="0" />\n' + \
           '        </attvalues>\n' + \
           '      </edge>\n' + \
           '    </edges>\n' + \
           '  </graph>\n' + \
           '</gexf>\n'
    print "Tests done."
