import torch
import numpy as np
from sklearn.metrics import f1_score, average_precision_score, mean_absolute_error
from bert_encoder import BERT_MultiLabelClassifier
from gnn_model import GNN_MusicClassifier
from fusion_model import GNN_BERT_Fusion
from contrastive import ContrastiveDualEncoder, evaluate_retrieval

def calculate_classification_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Computes Macro-F1, Micro-F1, and mean AUC-PR for multi-label tag prediction.
    """
    y_pred_binary = (y_pred_probs > threshold).astype(int)
    
    macro_f1 = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    # average_precision_score computes AUC-PR per class. We take the unweighted mean over tags.
    auc_pr = average_precision_score(y_true, y_pred_probs, average='macro')
    
    return {"Macro-F1": macro_f1, "Micro-F1": micro_f1, "AUC-PR": auc_pr}

def calculate_regression_metrics(v_true, v_pred, a_true, a_pred):
    """
    Computes MAE for DEAM valence and arousal regression.
    """
    mae_v = mean_absolute_error(v_true, v_pred)
    mae_a = mean_absolute_error(a_true, a_pred)
    
    return {"MAE_Valence": mae_v, "MAE_Arousal": mae_a}

def evaluate_models(model_weights_path='../results/model_weights.pt'):
    """
    Placeholder for evaluating the saved models on the test split.
    """
    print(f"Loading weights from {model_weights_path}...")
    
    # 1. Initialize models (assuming definitions exist in the other files)
    # bert = BERT_MultiLabelClassifier(num_tags=50)
    # gnn = GNN_MusicClassifier(in_channels=128, hidden_channels=64, num_classes=50)
    # fusion = GNN_BERT_Fusion(...)
    # contrastive = ContrastiveDualEncoder(...)
    
    # 2. Load test set labels and predictions (dummy data for scaffolding)
    # Replace with dataloader forward passes
    print("Running inference on test split...")
    
    num_samples = 100
    num_tags = 50
    
    # Dummy data
    y_true_tags = np.random.randint(0, 2, size=(num_samples, num_tags))
    y_pred_probs = np.random.rand(num_samples, num_tags)
    
    v_true = np.random.uniform(1, 9, size=(num_samples,))
    v_pred = np.random.uniform(1, 9, size=(num_samples,))
    a_true = np.random.uniform(1, 9, size=(num_samples,))
    a_pred = np.random.uniform(1, 9, size=(num_samples,))
    
    # 3. Calculate Task 1-3 Metrics
    print("\n--- Tasks 1-3 Evaluation ---")
    cls_metrics = calculate_classification_metrics(y_true_tags, y_pred_probs)
    print(f"Classification -> Macro-F1: {cls_metrics['Macro-F1']:.4f}, Micro-F1: {cls_metrics['Micro-F1']:.4f}, AUC-PR: {cls_metrics['AUC-PR']:.4f}")
    
    reg_metrics = calculate_regression_metrics(v_true, v_pred, a_true, a_pred)
    print(f"Emotion (DEAM) -> MAE Valence: {reg_metrics['MAE_Valence']:.4f}, MAE Arousal: {reg_metrics['MAE_Arousal']:.4f}")
    
    # 4. Calculate Task 4 Metrics
    print("\n--- Task 4 Contrastive Retrieval Evaluation ---")
    # S_matrix represents the graph <-> caption similarity matrix
    S_matrix = torch.rand(num_samples, num_samples) 
    
    # Add a diagonal boost to simulate a partially trained network
    S_matrix += torch.eye(num_samples) * 0.5 
    
    retrieval_metrics = evaluate_retrieval(S_matrix)
    
    print("Caption -> Audio (Querying audio with text):")
    print(f"  R@1:  {retrieval_metrics['Caption_to_Audio']['R@1']:.4f}")
    print(f"  R@5:  {retrieval_metrics['Caption_to_Audio']['R@5']:.4f}")
    print(f"  R@10: {retrieval_metrics['Caption_to_Audio']['R@10']:.4f}")
    
    print("Audio -> Caption (Querying text with audio):")
    print(f"  R@1:  {retrieval_metrics['Audio_to_Caption']['R@1']:.4f}")
    print(f"  R@5:  {retrieval_metrics['Audio_to_Caption']['R@5']:.4f}")
    print(f"  R@10: {retrieval_metrics['Audio_to_Caption']['R@10']:.4f}")

if __name__ == "__main__":
    evaluate_models()
