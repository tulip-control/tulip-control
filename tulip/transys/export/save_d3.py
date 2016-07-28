# Copyright (c) 2013 by California Institute of Technology
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
Export html containing d3.js animating SVG
"""
# there exists: https://github.com/mikedewar/d3py,
# but it is not sufficiently developed yet,
# so here the wheel is partially re-invented

import os
import inspect

from networkx.readwrite import json_graph

def _format_label(label_def, label_dot_format):
    """Format state/edge labels, which pop-up on mouse hover.
    """
    s = '"\\n\\n" +'
    for sublabel_name in label_def:
        shown_name = label_dot_format[sublabel_name]
        kv_sep = label_dot_format['type?label']
        sep = label_dot_format['separator']

        s += '"' +shown_name +kv_sep +'" '
        s += '+d.' +str(sublabel_name) +'+"' +sep+'" +'
    s += '" ";'

    return s

def labeled_digraph2d3(graph, html_file_name='index.html'):
    """Export to SVG embedded in HTML, animated with d3.js

    Example
    =======
    From C{examples/transys/machine_examples.py} call:

    >>> m = garage_counter_with_state_vars()

    Then export to html:

    >>> m.save('index.html', 'html')

    See Also
    ========
    FSM, BA, Mealy

    @param graph: labeled graph to export
    @type graph: L{LabeledDiGraph}
    """
    file_path = inspect.getfile(inspect.currentframe())
    dir_path = os.path.dirname(os.path.abspath(file_path) )

    d3_file_name = os.path.join(dir_path, 'd3.v3.min.js')
    d3_file = open(d3_file_name)
    d3_js = d3_file.read()

    s = """
    <!DOCTYPE html>
    <meta charset="utf-8">
    <style>

    .node {
      stroke: black;
      stroke-width: 1.5px;
    }

    .link {
      stroke: #999;
      stroke-opacity: .6;
    }

    .end-arrow {
        fill            : gray;
        stroke-width    : 1px;
    }

    </style>

    <script>
    """

    # embed d3.js to create single .html,
    # instead of bunch of files
    s += d3_js

    s += """
    </script>
    <body>

    <script>
    var width = 960,
        height = 500;

    var color = d3.scale.category20();

    var force = d3.layout.force()
        .charge(-120)
        .linkDistance(200)
        .size([width, height]);

    var svg = d3.select("body").append("svg")
        .attr("width", width)
        .attr("height", height);

    svg.append('svg:defs').append('svg:marker')
        .attr('id', 'end-arrow')
        .attr('viewBox', '0 -5 10 10')
        .attr('refX', 15)
        .attr('refY', 0)
        .attr('markerWidth', 4)
        .attr('markerHeight', 4)
        .attr('orient', 'auto')
      .append('svg:path')
        .attr('d', 'M0,-5L10,0L0,5')
        .attr('class', 'end-arrow');

    var graph = """

    # embed to avoid browser local file-loading restrictions
    try:
        s += json_graph.dumps(graph)
    except:
        # better error msg for numpy array
        import json
        data = json_graph.node_link_data(graph)
        s += json.dumps(data, default=lambda x: str(x) )

    s += ';'

    s += """
    function draw(graph){
      force
          .nodes(graph.nodes)
          .links(graph.links)
          .start();

      var link = svg.append("svg:g").selectAll("path")
          .data(graph.links)
        .enter().append("svg:path")
          .attr("class", "link")
          .style("stroke-width", 10)
          .style("fill", "none")
          .style("marker-end", 'url(#end-arrow)');

      link.append("title")
          .text(function(d) {
          	return """

    # edge labels (shown when mouse on edge)
    if hasattr(graph, '_transition_label_def') and \
    hasattr(graph, '_transition_dot_label_format'):
        transition_label_def = graph._transition_label_def
        transition_label_format = graph._transition_dot_label_format
        s += _format_label(transition_label_def, transition_label_format)
    else:
        s += '" ";'

    s += """});

      var node = svg.selectAll(".node")
          .data(graph.nodes)
       .enter().append("g")
          .attr("class", "node")
          .call(force.drag);

      node.append("circle")
          .attr("r", 30)
          .style("fill", "#66CC00")

      node.append("text")
          .attr("dx", 0)
          .attr("dy", 0)
          .attr("fill", "red")
          .text(function(d) { return d.id});

      node.append("title")
          .style("fill", "gray")
          .text(function(d) { return """

    # edge labels (shown when mouse on edge)
    if hasattr(graph, '_state_label_def') and \
    hasattr(graph, '_state_dot_label_format'):
        state_label_def = graph._state_label_def
        state_label_format = graph._state_dot_label_format
        s += _format_label(state_label_def, state_label_format)
    else:
        s += '" ";'

    s += """});

      force.on("tick", function() {
        link.attr("d", function(d) {
            var dx = d.target.x -d.source.x,
                dy = d.target.y -d.source.y,
                dr = Math.sqrt(dx * dx + dy * dy);
            return "M" +
                d.source.x + "," +
                d.source.y + "A" +
                dr + "," + dr + " 0 0,1 " +
                d.target.x + "," +
                d.target.y;
        });

        node.attr("cx", function(d) { return d.x; })
            .attr("cy", function(d) { return d.y; })
            .attr("transform", function(d) {
                return "translate(" + d.x + "," + d.y + ")";
            });
      });
    };

    draw(graph)

    </script>
    </body>
    """

    html_file = open(html_file_name, 'w')
    html_file.write(s)
    return True
