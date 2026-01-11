import torch
import torch.nn as nn
import torch.optim as optim
from model import HybridGNN
import os

def load_mock_data(path):
    """
    Loads the .pt graph file with security check disabled for local files.
    """
    if os.path.exists(path):
        # Allow loading custom objects (weights_only=False) since we trust our own file
        return torch.load(path, weights_only=False) 
    else:
        raise FileNotFoundError(f"Graph file not found at {path}. Did you run graph_builder.py?")

def train_one_epoch():
    print("--- STARTING TRAINING DRY RUN (Laptop Mode) ---")
    
    # 1. Setup Configuration
    model = HybridGNN(num_node_features=1, hidden_dim=128, text_dim=384, num_classes=4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Load Data
    graph_path = "data/processed_graphs/video_001.pt"
    try:
        data = load_mock_data(graph_path)
        
        # --- THE FIX IS HERE ---
        # Convert integer features to Float (decimal) so the Neural Network can read them
        data.x = data.x.float()
        
        print(f"Loaded Graph: {data}")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    # 3. Simulate Inputs
    mock_context_embedding = torch.randn(1, 384) 
    mock_target = torch.tensor([2], dtype=torch.long)
    batch_vector = torch.zeros(data.num_nodes, dtype=torch.long)

    # 4. The Training Loop
    model.train()
    print("\nTraining Loop Verification:")
    
    for step in range(1, 6):
        optimizer.zero_grad() 
        
        # Forward Pass
        output = model(data.x, data.edge_index, batch_vector, mock_context_embedding)
        
        loss = criterion(output, mock_target)
        loss.backward()
        optimizer.step()
        
        print(f"  Epoch {step}: Loss = {loss.item():.4f}")

    # 5. Save the Model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/pilot_model.pth")
    print("\nSUCCESS: Model trained and saved to 'checkpoints/pilot_model.pth'")

if __name__ == "__main__":
    train_one_epoch()