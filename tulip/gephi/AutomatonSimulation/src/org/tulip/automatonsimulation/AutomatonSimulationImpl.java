/* 
 * Copyright (c) 2011 by California Institute of Technology
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the California Institute of Technology nor
 *    the names of its contributors may be used to endorse or promote
 *    products derived from this software without specific prior
 *    written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
 * OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
package org.tulip.automatonsimulation;

import java.awt.Color;
import java.io.IOException;
import java.net.URL;
import org.openide.util.Lookup;
import org.gephi.project.api.ProjectController;
import org.gephi.project.api.Workspace;
import org.gephi.graph.api.GraphController;
import org.gephi.graph.api.GraphModel;
import org.gephi.graph.api.Node;
import org.gephi.graph.api.Edge;
import org.gephi.graph.api.HierarchicalDirectedGraph;
import org.gephi.streaming.api.StreamingController;
import org.gephi.streaming.api.StreamingConnection;
import org.gephi.streaming.api.StreamingEndpoint;
import org.gephi.data.attributes.api.AttributeController;
import org.gephi.data.attributes.api.AttributeModel;
import org.gephi.ranking.api.RankingController;
import org.gephi.ranking.api.Ranking;
import org.gephi.ranking.api.ColorTransformer;
import org.gephi.layout.plugin.forceAtlas.ForceAtlas;
import org.gephi.layout.plugin.forceAtlas.ForceAtlasLayout;

/*
 * Contains methods to access the Gephi API, for graph loading,
 * manipulation, and visualization.
 */
public class AutomatonSimulationImpl extends Thread
{
    /*
     * Fields
     */
    // When 'stop' changes to true, attempts to close all running methods.
    public volatile boolean stop = false;
    // 'delay' sets the number of ms to wait between rank updates and
    // automaton open/close checks.
    public int delay = 700;
    
    // Set active attribute name.
    public String activeID = "is_active";
    // Set setup parameters.
    public double labelSize = 0.7;
    // Set streaming parameters.
    public String streamURL =
          "http://localhost:8080/workspace0?operation=getGraph";
    // Set ranking parameters.
    public int startColor = 0xBBBBBB;
    public int endColor = 0x23C20E;
    // Set ForceAtlas parameters.
    public double repulsionStrength = 1000.0;
    public double attractionStrength = 1.0;
    public double gravity = 250.0;
    
    // Gephi controllers, models, and data.
    private ProjectController pc;
    private Workspace workspace;
    private GraphModel graphModel;
    private StreamingController streamingController;
    private AttributeModel attributeModel;
    private RankingController rankingController;
    private HierarchicalDirectedGraph graph;
    private Node[] automata;
    private Node[][] childrenList;
    private Edge[] edges;
    
    
    
    /* 
     * Constructor
     */
    AutomatonSimulationImpl()
    {
        try
        {
            // Get basic controllers and models.
            pc = Lookup.getDefault().lookup(ProjectController.class);
            workspace = pc.getCurrentWorkspace();
            graphModel = Lookup.getDefault().lookup(GraphController.class).getModel();
            streamingController = Lookup.getDefault().lookup(StreamingController.class);
            attributeModel = Lookup.getDefault().lookup(AttributeController.class).getModel();
            rankingController = Lookup.getDefault().lookup(RankingController.class);
            graph = graphModel.getHierarchicalDirectedGraph();
            automata = graph.getTopNodes().toArray();
            childrenList = new Node[automata.length][];
            for (int i=0; i<automata.length; i++)
            {
                // The 'i'th automaton has children in the 'i'th children list.
                childrenList[i] = graph.getChildren(automata[i]).toArray();
            }
            edges = graph.getEdges().toArray();
        }
        catch (NullPointerException e)
        {
            System.out.println("Oh, no! NullPointerException thrown...");
            e.printStackTrace();
        }
    }
    
    
    
    /* 
     * Delete all nodes and edges in the current graph.
     */
    public void deleteGraph() throws InterruptedException
    {
        graph.clear();
    }
    
    
    
    /* 
     * Prepare graph for simulation by expanding automata and configuring
     * label size.
     */
    public void setupGraph() throws InterruptedException
    {
        for (int i=0; i<automata.length; i++)
        {
            // Expand all automata to children.
            graph.expand(automata[i]);
            
            // Configure label size.
            for (int j=0; j<childrenList[i].length; j++)
            {
                childrenList[i][j].getNodeData().getTextData().
                      setSize((float) labelSize);
            }
        }
    }
    
    
    
    /* 
     * Start graph streaming from the streaming URL.
     */
    public void startStreaming() throws InterruptedException
    {
        try
        {
            StreamingEndpoint streamingEndpoint = new StreamingEndpoint(
                  new URL(streamURL),
                  streamingController.getStreamType("JSON"));
            StreamingConnection streamingConnection = streamingController.
                  connect(streamingEndpoint, graph);
            streamingConnection.asynchProcess();

            // Accept stream until signalled to stop.
            while (!stop) {}
            streamingConnection.close();
        }
        catch (IOException e)
        {
            System.out.println("Oh, no! IOException thrown...");
            e.printStackTrace();
        }
    }
    
    
    
    /* 
     * Perpetually expand 'active' automata and retract not 'active' automata.
     */
    public void viewActiveAutomata() throws InterruptedException
    {
        // Run until signalled to stop.
        while (!stop)
        {
            Thread.sleep(delay);
            for (int i=0; i<automata.length; i++)
            {
                // Check children for a nonzero active value.
                boolean activeChildren = false;
                for (int j=0; j<childrenList[i].length; j++)
                {
                    if (((Number) childrenList[i][j].getNodeData().
                          getAttributes().getValue(activeID)).intValue() != 0)
                    {
                        activeChildren = true;
                        break;
                    }
                }
                
                // Retract and expand automata as appropriate.
                if (!activeChildren && !graph.isInView(automata[i]))
                {
                    graph.retract(automata[i]);
                }
                else if (activeChildren && graph.isInView(automata[i]))
                {
                    graph.expand(automata[i]);
                }
            }
        }
    }
    
    
    
    /* 
     * Perpetually rank and color nodes by their 'active' attribute.
     */
    public void rankNodes() throws InterruptedException
    {
        // Set up node color ranking.
        Ranking activeRanking = rankingController.getRankingModel().
              getNodeAttributeRanking(attributeModel.getNodeTable().
              getColumn(activeID));
        ColorTransformer activeTransformer =
              rankingController.getObjectColorTransformer(activeRanking);
        activeTransformer.setColors(new Color[]
              {new Color(startColor), new Color(endColor)});
        
        // Rank nodes until signalled to stop.
        while (!stop)
        {
            Thread.sleep(delay);
            rankingController.transform(activeTransformer);
        }
    }
    
    
    
    /* 
     * Perpetually use a Force Atlas layout on the currently visible graph.
     */
    public void layoutGraph() throws InterruptedException
    {
        // Set up Force Atlas layout.
        ForceAtlasLayout forceAtlasLayout = new ForceAtlas().buildLayout();
        forceAtlasLayout.setGraphModel(graphModel);
        forceAtlasLayout.resetPropertiesValues();
        forceAtlasLayout.setRepulsionStrength(repulsionStrength);
        forceAtlasLayout.setAttractionStrength(attractionStrength);
        forceAtlasLayout.setGravity(gravity);
        
        // Run Force Atlas while possible.
        while (!stop && forceAtlasLayout.canAlgo())
        {
            forceAtlasLayout.goAlgo();
        }
    }
}