import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyConcatFusion(nn.Module):
    def __init__(self, gnn_hidden_dim, bert_hidden_dim, num_tags):
        super(EarlyConcatFusion, self).__init__()
        
        concat_dim = gnn_hidden_dim + bert_hidden_dim
        
        # Classification head for tags
        self.tag_classifier = nn.Linear(concat_dim, num_tags)
        
        # Regression heads for Valence and Arousal (DEAM multi-task)
        self.valence_regressor = nn.Linear(concat_dim, 1)
        self.arousal_regressor = nn.Linear(concat_dim, 1)
        
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"EarlyConcatFusion initialized with {trainable_params:,} trainable parameters")

    def forward(self, g, H_text):
        """
        g: Graph-level representation [batch_size, gnn_hidden_dim]
        H_text: BERT token embeddings [batch_size, seq_len, bert_hidden_dim]
        """
        # For early concat, we just use the CLS token (first token) from BERT
        cls_token = H_text[:, 0, :]
        
        # z = CONCAT(g, t)
        z = torch.cat([g, cls_token], dim=-1)
        
        # Predictions
        y_hat_tags = self.tag_classifier(z)
        v_hat = self.valence_regressor(z).squeeze(-1)
        a_hat = self.arousal_regressor(z).squeeze(-1)
        
        return y_hat_tags, v_hat, a_hat

class GNN_BERT_Fusion(nn.Module):
    def __init__(self, gnn_hidden_dim, bert_hidden_dim, num_tags, d_k=64):
        """
        Cross-attention fusion between structural (GNN) and semantic text (BERT) representations.
        gnn_hidden_dim: Dimension of the GNN graph readout 'g'.
        bert_hidden_dim: Dimension of the BERT contextual embeddings 'H_{text}'.
        d_k: Hidden dimension for query/key projections.
        """
        super(GNN_BERT_Fusion, self).__init__()
        
        self.d_k = d_k
        self.W_Q = nn.Linear(gnn_hidden_dim, d_k, bias=False)
        self.W_K = nn.Linear(bert_hidden_dim, d_k, bias=False)
        
        # Classification head for tags
        concat_dim = gnn_hidden_dim + bert_hidden_dim
        self.tag_classifier = nn.Linear(concat_dim, num_tags)
        
        # Regression heads for Valence and Arousal (DEAM multi-task)
        self.valence_regressor = nn.Linear(concat_dim, 1)
        self.arousal_regressor = nn.Linear(concat_dim, 1)

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GNN_BERT_Fusion (Cross-Attention) initialized with {trainable_params:,} trainable parameters")

    def forward(self, g, H_text):
        """
        g: Graph-level representation [batch_size, gnn_hidden_dim]
        H_text: BERT token embeddings [batch_size, seq_len, bert_hidden_dim]
        """
        # Q = g * W_Q (Shape: [batch_size, 1, d_k])
        # We add unsqueeze to align dimensions for bmm
        Q = self.W_Q(g).unsqueeze(1)
        
        # K = H_text * W_K (Shape: [batch_size, seq_len, d_k])
        K = self.W_K(H_text)
        
        # Attention scores A = softmax(QK^T / sqrt(d))
        # K.transpose(1, 2) shape: [batch_size, d_k, seq_len]
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)
        A = F.softmax(scores, dim=-1) # Shape: [batch_size, 1, seq_len]
        
        # AH_{text} (Shape: [batch_size, 1, bert_hidden_dim])
        attn_out = torch.bmm(A, H_text).squeeze(1) # [batch_size, bert_hidden_dim]
        
        # z = CONCAT(g, AH_{text})
        z = torch.cat([g, attn_out], dim=-1)
        
        # Predictions
        y_hat_tags = self.tag_classifier(z) # Logits for tags
        v_hat = self.valence_regressor(z).squeeze(-1) # Valence prediction
        a_hat = self.arousal_regressor(z).squeeze(-1) # Arousal prediction
        
        return y_hat_tags, v_hat, a_hat

class MultiTaskFusionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, multi_label=False):
        super(MultiTaskFusionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if multi_label:
            self.tag_loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.tag_loss_fn = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_hat_tags, y_tags, v_hat=None, v=None, a_hat=None, a=None):
        """
        Multi-task loss: L = L_tags + alpha * ||v - v_hat||^2 + beta * ||a - a_hat||^2
        """
        L_tags = self.tag_loss_fn(y_hat_tags, y_tags)
        
        # Compute MSE only if valence/arousal targets are available
        if v is not None and a is not None and v_hat is not None and a_hat is not None:
            L_v = self.mse_loss(v_hat, v)
            L_a = self.mse_loss(a_hat, a)
            return L_tags + self.alpha * L_v + self.beta * L_a
            
        return L_tags
