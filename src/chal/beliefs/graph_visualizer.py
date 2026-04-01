"""
graph_visualizer.py

Interactive HTML visualization of belief graphs using Cytoscape.js.

Generates standalone HTML files that allow users to:
- View argument structure as interactive node-edge graph
- Click on nodes to see details (claims, evidence, assumptions)
- Explore questions/rebuttals targeting specific nodes
- Track how beliefs evolved through the debate
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
from chal.beliefs.belief_graph import BeliefGraph


def export_debate_graph(
    agents: List[Any],
    topic: str,
    challenge_rebuttal_pairs: List[Dict],
    output_path: Path
) -> str:
    """
    Generate interactive HTML visualization of entire debate.

    Args:
        agents: List of Agent objects with their beliefs
        topic: Debate topic
        challenge_rebuttal_pairs: All Q&A exchanges from the debate
        output_path: Where to save the HTML file

    Returns:
        HTML content as string
    """
    print(f"      [Graph] Collecting beliefs from {len(agents)} agents...")
    # Collect all belief snapshots from all agents
    agent_beliefs = []
    for agent in agents:
        try:
            belief_obj = agent.get_internal_belief_obj()
            if belief_obj:
                agent_beliefs.append({
                    "agent_name": agent.name,
                    "belief": belief_obj
                })
        except Exception as e:
            print(f"Warning: Could not get belief from {agent.name}: {e}")
            continue

    print(f"      [Graph] Building graph data for {len(agent_beliefs)} agent belief(s)...")
    # Build graph data for Cytoscape
    nodes = []
    edges = []
    node_id_counter = 0

    # Color scheme by node type
    node_colors = {
        "assumption": "#3498db",  # Blue
        "claim": "#e74c3c",       # Red
        "evidence": "#2ecc71",    # Green
        "prediction": "#f39c12"   # Orange
    }

    # Process each agent's belief
    for agent_data in agent_beliefs:
        agent_name = agent_data["agent_name"]
        belief = agent_data["belief"]

        print(f"      [Graph] Processing {agent_name}...")
        try:
            graph = BeliefGraph(belief)
            print(f"      [Graph] Built graph for {agent_name}: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        except Exception as e:
            print(f"Warning: Could not build graph for {agent_name}: {e}")
            continue

        # Add nodes from this agent's belief
        for node_id, node_info in graph.nodes.items():
            node_type = node_info["type"]
            node_data = node_info["data"]

            # Determine label
            if node_type == "claim":
                label = f"{node_id}: {node_data.get('statement', '')[:50]}..."
            elif node_type == "assumption":
                label = f"{node_id}: {node_data.get('statement', '')[:50]}..."
            elif node_type == "evidence":
                label = f"{node_id}: {node_data.get('summary', '')[:50]}..."
            elif node_type == "prediction":
                label = f"{node_id}: {node_data.get('statement', '')[:50]}..."
            else:
                label = node_id

            try:
                # Try to serialize node_data, truncate if too large
                details_str = json.dumps(node_data, indent=2, ensure_ascii=False)
                if len(details_str) > 10000:  # Truncate very large nodes
                    details_str = details_str[:10000] + "\n... (truncated)"
            except Exception as e:
                details_str = f"{{\"error\": \"Could not serialize node data: {str(e)}\"}}"

            nodes.append({
                "data": {
                    "id": f"{agent_name}_{node_id}",
                    "label": label,
                    "node_type": node_type,
                    "agent": agent_name,
                    "details": details_str,
                    "color": node_colors.get(node_type, "#95a5a6")
                }
            })

        # Add edges from this agent's belief
        for from_id, to_id, edge_type in graph.edges:
            edges.append({
                "data": {
                    "source": f"{agent_name}_{from_id}",
                    "target": f"{agent_name}_{to_id}",
                    "edge_type": edge_type
                }
            })

    print(f"      [Graph] Total nodes: {len(nodes)}, Total edges: {len(edges)}")
    print(f"      [Graph] Building Q&A overlay data...")
    # Build Q&A overlay data
    qa_data = []
    for pair in challenge_rebuttal_pairs:
        challenger = pair.get("challenger", "Unknown")
        target = pair.get("target", "Unknown")
        question = pair.get("challenge", "")
        qid = pair.get("qid", "?")
        target_ids = pair.get("target_ids", [])
        rebuttal = pair.get("rebuttal", "")
        resolution = pair.get("resolution", {})

        qa_data.append({
            "challenger": challenger,
            "target": target,
            "qid": qid,
            "question": question,
            "target_ids": target_ids,
            "attack_type": pair.get("attack_type", ""),
            "attack_strategy": pair.get("attack_strategy", ""),
            "rebuttal": rebuttal,
            "resolution": resolution
        })

    print(f"      [Graph] Generating HTML output...")

    # Pre-serialize JSON data to avoid hanging in f-string
    print(f"      [Graph] Serializing {len(nodes)} nodes...")
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    print(f"      [Graph] Serializing {len(edges)} edges...")
    edges_json = json.dumps(edges, ensure_ascii=False)
    print(f"      [Graph] Serializing {len(qa_data)} Q&A pairs...")
    qa_json = json.dumps(qa_data, ensure_ascii=False)

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CHAL Debate Graph - {topic}</title>
    <script src="https://unpkg.com/cytoscape@3.26.0/dist/cytoscape.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            overflow: hidden;
        }}

        #container {{
            display: flex;
            height: 100vh;
        }}

        #cy {{
            flex: 1;
            background: #2a2a2a;
        }}

        #sidebar {{
            width: 400px;
            background: #1e1e1e;
            border-left: 1px solid #333;
            overflow-y: auto;
            padding: 20px;
        }}

        #sidebar h2 {{
            color: #3498db;
            margin-bottom: 15px;
            font-size: 20px;
        }}

        #sidebar h3 {{
            color: #e74c3c;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 16px;
        }}

        #sidebar pre {{
            background: #2a2a2a;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.4;
        }}

        .legend {{
            margin-bottom: 20px;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 4px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 10px;
        }}

        .qa-item {{
            background: #2a2a2a;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 3px solid #3498db;
        }}

        .qa-item strong {{
            color: #3498db;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="cy"></div>
        <div id="sidebar">
            <h2>Debate: {topic}</h2>

            <div class="legend">
                <h3>Legend</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background: #3498db;"></div>
                    <span>Assumption (A#)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #e74c3c;"></div>
                    <span>Claim (C#)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2ecc71;"></div>
                    <span>Evidence (E#)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f39c12;"></div>
                    <span>Prediction (P#)</span>
                </div>
            </div>

            <div id="details">
                <p style="color: #888;">Click on a node to see details</p>
            </div>
        </div>
    </div>

    <script>
        const graphData = {{
            nodes: {nodes_json},
            edges: {edges_json}
        }};

        const qaData = {qa_json};

        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: [
                ...graphData.nodes,
                ...graphData.edges
            ],
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'background-color': 'data(color)',
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'font-size': '12px',
                        'color': '#ffffff',
                        'text-outline-color': '#000000',
                        'text-outline-width': 2,
                        'width': 80,
                        'height': 80
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 2,
                        'line-color': '#666',
                        'target-arrow-color': '#666',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier'
                    }}
                }}
            ],
            layout: {{
                name: 'breadthfirst',
                directed: true,
                padding: 30,
                spacingFactor: 1.5
            }}
        }});

        // Node click handler
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const nodeId = node.data('id');
            const agent = node.data('agent');
            const nodeType = node.data('node_type');
            const details = node.data('details');

            // Find Q&A targeting this node
            const rawId = nodeId.replace(agent + '_', '');
            const relevantQA = qaData.filter(qa =>
                qa.target === agent && qa.target_ids && qa.target_ids.includes(rawId)
            );

            // Build details HTML
            let html = `<h2>${{nodeId}}</h2>`;
            html += `<p><strong>Type:</strong> ${{nodeType}}</p>`;
            html += `<p><strong>Agent:</strong> ${{agent}}</p>`;
            html += `<h3>Details</h3>`;
            html += `<pre>${{details}}</pre>`;

            if (relevantQA.length > 0) {{
                html += `<h3>Questions Targeting This Node (${{relevantQA.length}})</h3>`;
                relevantQA.forEach(qa => {{
                    html += `<div class="qa-item">`;
                    html += `<p><strong>${{qa.qid}}</strong> from <strong>${{qa.challenger}}</strong></p>`;
                    if (qa.attack_type) {{
                        html += `<p style="color: #888; font-size: 0.9em;"><em>${{qa.attack_type}} / ${{qa.attack_strategy}}</em></p>`;
                    }}
                    html += `<p>${{qa.question}}</p>`;
                    if (qa.rebuttal) {{
                        html += `<p style="margin-top: 8px;"><em>Rebuttal:</em> ${{qa.rebuttal}}</p>`;
                    }}
                    html += `</div>`;
                }});
            }}

            document.getElementById('details').innerHTML = html;
        }});
    </script>
</body>
</html>"""

    # Write to file
    print(f"      [Graph] Writing to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"      [Graph] Done!")
    return html
