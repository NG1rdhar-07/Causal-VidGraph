import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class HybridGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim=128, text_dim=384, num_classes=4):
        super(HybridGNN, self).__init__()
        
        # --- Path A: Graph Reasoning (The Logic) ---
        # GAT (Graph Attention Network) allows the model to focus on important edges.
        # SRS Requirement: 3 Layers, Hidden Dim 128 [cite: 59]
        self.gat1 = GATConv(num_node_features, hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim)
        self.gat3 = GATConv(hidden_dim, hidden_dim)
        
        # --- Path B: Context Encoder (The Semantic Safety Net) ---
        # We assume the text is already embedded into a vector (e.g., by BERT)
        # This layer processes that text vector to match the graph dimension.
        self.text_processor = nn.Linear(text_dim, hidden_dim)
        
        # --- Fusion Layer ---
        # Concatenates Graph features + Text features
        fusion_dim = hidden_dim + hidden_dim 
        
        # --- Classifier ---
        # MLP: Linear -> ReLU -> Linear -> Output
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5), # Prevent overfitting
            nn.Linear(64, num_classes)
        )

    def forward(self, x, edge_index, batch, context_embedding):
        """
        x: Node features (from graph)
        edge_index: Connectivity (who is connected to whom)
        batch: Keeps track of which node belongs to which video (for batching)
        context_embedding: The vector from the text description (BERT)
        """
        
        # 1. Graph Path (GAT)
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        x = self.gat3(x, edge_index)
        
        # 2. Pooling (Graph -> Single Vector)
        # Summarize the whole graph into one vector representing the "Video Logic"
        graph_features = global_mean_pool(x, batch)  # Shape: [batch_size, hidden_dim]
        
        # 3. Text Path
        # Process the text context
        text_features = self.text_processor(context_embedding) # Shape: [batch_size, hidden_dim]
        text_features = F.relu(text_features)
        
        # 4. Fusion
        # Combine Logic (Graph) + Context (Text)
        combined = torch.cat([graph_features, text_features], dim=1)
        
        # 5. Prediction
        output = self.classifier(combined)
        
        return output

# --- Simple Test Block ---
if __name__ == "__main__":
    # This runs only if you run 'python src/model.py' directly
    print("Testing HybridGNN Architecture...")
    
    # Mock Data
    num_nodes = 11
    node_features = 1 # We used simple hashes (1 dim)
    
    # Initialize Model
    model = HybridGNN(num_node_features=node_features)
    print("Model Created Successfully!")
    print(model)
    
    # Create Dummy Inputs to check dimensions
    x = torch.randn(num_nodes, node_features) # Random node features
    edge_index = torch.randint(0, num_nodes, (2, 22)) # Random edges
    batch = torch.zeros(num_nodes, dtype=torch.long) # All nodes belong to video 0
    context = torch.randn(1, 384) # Fake BERT embedding
    
    # Forward Pass
    try:
        out = model(x, edge_index, batch, context)
        print(f"\nForward Pass Successful! Output Shape: {out.shape}")
        print("Model is ready for training.")
    except Exception as e:
        print(f"\nForward Pass Failed: {e}")