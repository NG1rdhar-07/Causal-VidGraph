import json
import os
import argparse
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
import matplotlib.pyplot as plt

def build_graph(json_path):
    """
    Converts a Clean JSON video description into a Spatio-Temporal Graph.
    SRS Requirement: Use NetworkX -> PyTorch Geometric.
    """
    with open(json_path, 'r') as f:
        video_data = json.load(f)

    # 1. Initialize a directed graph
    G = nx.DiGraph()
    
    # We need to map (object_name, frame_id) to unique Node IDs
    # Format: "boy_frame0", "ball_frame0", etc.
    
    frames = video_data['frames']
    
    # --- Step A: Add Nodes & Spatial Edges (Within Frame) ---
    print(f"Building Graph for {video_data['video_id']}...")
    
    for frame in frames:
        frame_id = frame['frame_id']
        objs = frame['objects']
        
        # Add nodes for this frame
        for obj in objs:
            node_name = f"{obj}_{frame_id}"
            # We store the object name as a feature (converted to int hash for now)
            # In the real model, this would be a word embedding.
            obj_hash = hash(obj) % 1000  
            G.add_node(node_name, x=obj_hash, label=obj, time=frame_id)
        
        # Add Spatial Edges (fully connect objects in the same frame)
        # If 'boy' and 'ball' are in frame 0, connect them.
        import itertools
        for obj1, obj2 in itertools.combinations(objs, 2):
            u = f"{obj1}_{frame_id}"
            v = f"{obj2}_{frame_id}"
            G.add_edge(u, v, edge_type="spatial")
            G.add_edge(v, u, edge_type="spatial") # Bi-directional

    # --- Step B: Add Temporal Edges (Across Frames) ---
    # Connect "boy_frame0" -> "boy_frame1"
    
    for i in range(len(frames) - 1):
        curr_frame = frames[i]
        next_frame = frames[i+1]
        
        curr_objs = set(curr_frame['objects'])
        next_objs = set(next_frame['objects'])
        
        # Find objects that exist in both frames (Persistent objects)
        common_objs = curr_objs.intersection(next_objs)
        
        for obj in common_objs:
            u = f"{obj}_{curr_frame['frame_id']}"
            v = f"{obj}_{next_frame['frame_id']}"
            G.add_edge(u, v, edge_type="temporal")

    # --- Step C: Convert to PyTorch Geometric ---
    # PyG requires numeric features. NetworkX handles this conversion partially.
    # We assign a dummy feature vector for now to make it compatible.
    
    # Convert node attributes to numeric tensors
    for node in G.nodes():
        G.nodes[node]['x'] = [G.nodes[node]['x']] # Feature must be a list
        del G.nodes[node]['label'] # Remove string labels (PyG doesn't like strings)

    # Convert to PyG Data object
    pyg_data = from_networkx(G)
    
    print(f"  > Generated Graph: {pyg_data}")
    print(f"  > Nodes: {pyg_data.num_nodes}, Edges: {pyg_data.num_edges}")
    
    return pyg_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/processed_json')
    parser.add_argument('--output_dir', type=str, default='data/processed_graphs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process all JSON files in the input directory
    for filename in os.listdir(args.input_dir):
        if filename.endswith(".json"):
            input_path = os.path.join(args.input_dir, filename)
            
            try:
                graph_data = build_graph(input_path)
                
                # Save as .pt file (PyTorch standard)
                output_filename = filename.replace("_clean.json", ".pt")
                output_path = os.path.join(args.output_dir, output_filename)
                torch.save(graph_data, output_path)
                print(f"  > Saved to {output_path}\n")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()