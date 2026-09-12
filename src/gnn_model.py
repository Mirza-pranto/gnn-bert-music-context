import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GNN_MusicClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2):
        super(GNN_MusicClassifier, self).__init__()
        
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            
        # Graph readout (mean pooling) happens in forward pass
        # Classification head
        self.classifier = nn.Linear(hidden_channels, num_classes)
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GNN initialized with {trainable_params:,} trainable parameters")
        
    def forward(self, x, edge_index, batch, return_embeddings=False):

        # Message passing layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            
        # Graph readout (mean pooling)
        # g = MEANPOOL({h_i^(L)})
        g = global_mean_pool(x, batch)
        
        # Classification head (\hat{y} = \sigma(Wg + b))
        logits = self.classifier(g)
        
        if return_embeddings:
            return logits, g
            
        return logits
