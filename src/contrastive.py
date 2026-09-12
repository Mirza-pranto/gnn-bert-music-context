import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveDualEncoder(nn.Module):
    def __init__(self, gnn_encoder, bert_encoder, tau=0.07):
        """
        Dual-encoder architecture for cross-modal alignment between
        music graphs (GNN) and captions (BERT).
        
        gnn_encoder: Model returning graph embedding g_i
        bert_encoder: Model returning CLS token embedding t_i
        tau: Temperature parameter for InfoNCE loss
        """
        super(ContrastiveDualEncoder, self).__init__()
        self.gnn_encoder = gnn_encoder
        self.bert_encoder = bert_encoder
        self.tau = tau

    def forward(self, graph_data, input_ids, attention_mask):
        
        g = self.gnn_encoder(graph_data.x, graph_data.edge_index, graph_data.batch)
        g_norm = F.normalize(g, p=2, dim=-1)
        
        
        t = self.bert_encoder(input_ids, attention_mask)
        t_norm = F.normalize(t, p=2, dim=-1)
        
        
        S = torch.matmul(g_norm, t_norm.t()) / self.tau
        
        
        batch_size = S.size(0)
        labels = torch.arange(batch_size, device=S.device)
        
         for symmetry
        loss_g2t = F.cross_entropy(S, labels)
        loss_t2g = F.cross_entropy(S.t(), labels)
        loss_nce = (loss_g2t + loss_t2g) / 2.0
        
        return loss_nce, S

def evaluate_retrieval(S_matrix):

    N = S_matrix.size(0)
    
    # Audio -> Caption Retrieval (Rows)
    sorted_indices_a2c = torch.argsort(S_matrix, dim=1, descending=True)
    targets = torch.arange(N, device=S_matrix.device).view(-1, 1)
    
    a2c_ranks = (sorted_indices_a2c == targets).nonzero(as_tuple=True)[1]
    
    a2c_r1 = (a2c_ranks < 1).float().mean().item()
    a2c_r5 = (a2c_ranks < 5).float().mean().item()
    a2c_r10 = (a2c_ranks < 10).float().mean().item()
    
    # Caption -> Audio Retrieval (Columns)
    sorted_indices_c2a = torch.argsort(S_matrix, dim=0, descending=True).t()
    
    c2a_ranks = (sorted_indices_c2a == targets).nonzero(as_tuple=True)[1]
    
    c2a_r1 = (c2a_ranks < 1).float().mean().item()
    c2a_r5 = (c2a_ranks < 5).float().mean().item()
    c2a_r10 = (c2a_ranks < 10).float().mean().item()
    
    metrics = {
        "Audio_to_Caption": {"R@1": a2c_r1, "R@5": a2c_r5, "R@10": a2c_r10},
        "Caption_to_Audio": {"R@1": c2a_r1, "R@5": c2a_r5, "R@10": c2a_r10}
    }
    
    return metrics
