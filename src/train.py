import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
# --- FIX 1: Use the Graph-Specific DataLoader ---
from torch_geometric.loader import DataLoader 
from model import HybridGNN
import os
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
CSV_PATH = "data/next_qa/train.csv"
GRAPH_DIR = "data/processed_graphs"
MOCK_GRAPH_PATH = "data/processed_graphs/video_001.pt" 

class NextQADataset(Dataset):
    def __init__(self, csv_file, root_dir, mock_fallback=True):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mock_fallback = mock_fallback
        
        if self.mock_fallback and not os.path.exists(MOCK_GRAPH_PATH):
            raise FileNotFoundError(f"Mock graph not found at {MOCK_GRAPH_PATH}. Run graph_builder.py first!")

    def __len__(self):
        # Limit to 10 samples for laptop testing
        return min(len(self.annotations), 10) 

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_id = str(row['video'])
        label = int(row['answer']) 
        
        # Load Graph
        graph_path = os.path.join(self.root_dir, f"{video_id}.pt")
        
        if os.path.exists(graph_path):
            data = torch.load(graph_path, weights_only=False)
        elif self.mock_fallback:
            data = torch.load(MOCK_GRAPH_PATH, weights_only=False)
        else:
            raise FileNotFoundError(f"Graph {graph_path} not found.")

        # Ensure features are Float
        data.x = data.x.float()

        # --- FIX 2: Correct Dimensions for Text ---
        # Generate 1D vector (384). The Loader will stack them into [Batch_Size, 384]
        text_embedding = torch.randn(384) 

        return data, text_embedding, label

def train_epoch():
    print("--- STARTING TRAINING PIPELINE (PyG Loader Fixed) ---")
    
    # 1. Setup Dataset & Loader
    dataset = NextQADataset(csv_file=CSV_PATH, root_dir=GRAPH_DIR)
    
    # Batch Size 2: This now works because PyG Loader knows how to combine graphs!
    loader = DataLoader(dataset, batch_size=2, shuffle=True) 
    
    # 2. Setup Model
    model = HybridGNN(num_node_features=1, hidden_dim=128, text_dim=384, num_classes=5)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0
    
    # 3. Training Loop
    # The loader automatically unpacks the tuple from __getitem__
    for batch_idx, (graphs, text_embeddings, labels) in enumerate(loader):
        optimizer.zero_grad()
        
        # 'graphs' is now a special 'Batch' object containing multiple graphs
        # graphs.batch is a vector that tells the model which node belongs to which graph
        
        # Forward Pass
        output = model(graphs.x, graphs.edge_index, graphs.batch, text_embeddings)
        
        # Calculate Loss
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"  Batch {batch_idx+1}: Loss = {loss.item():.4f}")

    print(f"\nSUCCESS: Average Loss: {total_loss / len(loader):.4f}")
    
    # Save Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/pilot_model.pth")
    print("Model saved to 'checkpoints/pilot_model.pth'")

if __name__ == "__main__":
    train_epoch()