
import os
import sys
import json
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'data' / 'raw' / 'fma'))

from bert_encoder import BERT_MultiLabelClassifier, get_tokenizer
from gnn_model import GNN_MusicClassifier
from fusion_model import GNN_BERT_Fusion, EarlyConcatFusion, MultiTaskFusionLoss
from contrastive import ContrastiveDualEncoder, evaluate_retrieval
from data_loader import (
    get_fma_dataloaders,
    get_fma_text_dataloaders,
    get_fma_text_graph_dataloaders,
    get_deam_dataloaders,
)


def load_config(config_path='config.yaml'):
    if not os.path.isabs(config_path):
        config_path = project_root / config_path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)


# ============================================================
# TASK 1: BERT Genre Classifier
# ============================================================
def train_bert_task1(config, device):
    """Train BERT for single-label genre classification on FMA text descriptions."""
    print("\n" + "="*60)
    print("TASK 1: BERT Genre Classifier")
    print("="*60)
    
    cfg = config['task1_bert']
    
    # Load data
    print("Loading FMA text data...")
    train_loader, val_loader, test_loader, genre_to_idx = get_fma_text_dataloaders()
    num_classes = len(genre_to_idx)
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")
    print(f"  Classes: {num_classes} -> {genre_to_idx}")
    
    # Model
    model = BERT_MultiLabelClassifier(
        num_tags=num_classes,
        model_name=cfg['model_name'],
        freeze_bert=cfg['freeze_bert']
    ).to(device)
    
    tokenizer = get_tokenizer(cfg['model_name'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    metrics = {
        "train_loss": [], "val_loss": [],
        "macro_f1": [], "micro_f1": [], "accuracy": [],
        "wall_time_seconds": [],
    }
    
    t_total_start = time.time()
    best_val_f1 = 0.0
    
    for epoch in range(cfg['num_epochs']):
        t_epoch_start = time.time()
        
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        
        for batch_texts, batch_labels, _ in train_loader:
            encoded = tokenizer(
                list(batch_texts), padding=True, truncation=True,
                max_length=cfg['max_seq_length'], return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        avg_train_loss = epoch_loss / max(epoch_batches, 1)
        
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_texts, batch_labels, _ in val_loader:
                encoded = tokenizer(
                    list(batch_texts), padding=True, truncation=True,
                    max_length=cfg['max_seq_length'], return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                labels = batch_labels.to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_batches += 1
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_labels.numpy())
        
        avg_val_loss = val_loss / max(val_batches, 1)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        
        epoch_time = time.time() - t_epoch_start
        
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(avg_val_loss)
        metrics["macro_f1"].append(macro_f1)
        metrics["micro_f1"].append(micro_f1)
        metrics["accuracy"].append(acc)
        metrics["wall_time_seconds"].append(epoch_time)
        
        print(f"  Epoch {epoch+1}/{cfg['num_epochs']} -- "
              f"train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, "
              f"macro_f1: {macro_f1:.4f}, micro_f1: {micro_f1:.4f}, acc: {acc:.4f} "
              f"[{epoch_time:.1f}s]", flush=True)
        
        # Save best checkpoint
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            ckpt_path = str(project_root / config['paths']['checkpoints'] / 'task1_bert_best.pt')
            save_checkpoint(model, optimizer, epoch, metrics, ckpt_path)
            print(f"    -> Saved best checkpoint (macro_f1={macro_f1:.4f})")
    
    total_time = time.time() - t_total_start
    print(f"  Total wall time: {total_time:.1f}s ({total_time/60:.1f}min)")
    metrics["total_wall_time"] = total_time
    
    return metrics, model


# ============================================================
# TASK 2: GraphSAGE Genre Classifier
# ============================================================
def train_gnn_task2(config, device):
    """Train GraphSAGE for single-label genre classification on FMA audio graphs."""
    print("\n" + "="*60)
    print("TASK 2: GraphSAGE Genre Classifier")
    print("="*60)
    
    cfg = config['task2_gnn']
    
    # Load data
    print("Loading FMA graph data...")
    train_loader, val_loader, test_loader, genre_to_idx = get_fma_dataloaders()
    num_classes = len(genre_to_idx)
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")
    print(f"  Classes: {num_classes}")
    
    # Model
    model = GNN_MusicClassifier(
        in_channels=cfg['in_channels'],
        hidden_channels=cfg['hidden_channels'],
        num_classes=num_classes,
        num_layers=cfg['num_layers']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    metrics = {
        "train_loss": [], "val_loss": [],
        "macro_f1": [], "micro_f1": [], "accuracy": [],
        "wall_time_seconds": [],
    }
    
    t_total_start = time.time()
    best_val_f1 = 0.0
    
    for epoch in range(cfg['num_epochs']):
        t_epoch_start = time.time()
        
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        avg_train_loss = epoch_loss / max(epoch_batches, 1)
        
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
                
                val_loss += loss.item()
                val_batches += 1
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch.y.cpu().numpy())
        
        avg_val_loss = val_loss / max(val_batches, 1)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        
        epoch_time = time.time() - t_epoch_start
        
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(avg_val_loss)
        metrics["macro_f1"].append(macro_f1)
        metrics["micro_f1"].append(micro_f1)
        metrics["accuracy"].append(acc)
        metrics["wall_time_seconds"].append(epoch_time)
        
        print(f"  Epoch {epoch+1}/{cfg['num_epochs']} -- "
              f"train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, "
              f"macro_f1: {macro_f1:.4f}, micro_f1: {micro_f1:.4f}, acc: {acc:.4f} "
              f"[{epoch_time:.1f}s]", flush=True)
        
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            ckpt_path = str(project_root / config['paths']['checkpoints'] / 'task2_gnn_best.pt')
            save_checkpoint(model, optimizer, epoch, metrics, ckpt_path)
            print(f"    -> Saved best checkpoint (macro_f1={macro_f1:.4f})")
    
    total_time = time.time() - t_total_start
    print(f"  Total wall time: {total_time:.1f}s ({total_time/60:.1f}min)")
    metrics["total_wall_time"] = total_time
    
    return metrics, model


# ============================================================
# TASK 3: Cross-Attention Fusion (+ Early Concat Ablation)
# ============================================================
def train_fusion_task3(config, device, gnn_model=None, bert_model=None):
    """Train cross-attention fusion model on paired (graph, text) data."""
    print("\n" + "="*60)
    print("TASK 3: Cross-Attention Fusion")
    print("="*60)
    
    cfg = config['task3_fusion']
    
    # Load paired data
    print("Loading paired FMA text+graph data...")
    train_loader, val_loader, test_loader, genre_to_idx = get_fma_text_graph_dataloaders()
    num_classes = len(genre_to_idx)
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")
    
    # Load pretrained GNN if available
    if gnn_model is None:
        gnn_model = GNN_MusicClassifier(
            in_channels=config['task2_gnn']['in_channels'],
            hidden_channels=config['task2_gnn']['hidden_channels'],
            num_classes=num_classes,
            num_layers=config['task2_gnn']['num_layers']
        ).to(device)
        ckpt_path = project_root / config['paths']['checkpoints'] / 'task2_gnn_best.pt'
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), weights_only=False)
            gnn_model.load_state_dict(ckpt['model_state_dict'])
            print("  Loaded pretrained GNN checkpoint")
    
    # Load pretrained BERT if available
    if bert_model is None:
        bert_model = BERT_MultiLabelClassifier(
            num_tags=num_classes,
            model_name=config['task1_bert']['model_name'],
        ).to(device)
        ckpt_path = project_root / config['paths']['checkpoints'] / 'task1_bert_best.pt'
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), weights_only=False)
            bert_model.load_state_dict(ckpt['model_state_dict'])
            print("  Loaded pretrained BERT checkpoint")
    
    tokenizer = get_tokenizer(config['task1_bert']['model_name'])
    
    # Freeze encoders for fusion training  
    for p in gnn_model.parameters():
        p.requires_grad = False
    for p in bert_model.parameters():
        p.requires_grad = False
    gnn_model.eval()
    bert_model.eval()
    
    # Results for both fusion methods
    all_results = {}
    
    for fusion_type in ['cross_attention', 'early_concat']:
        print(f"\n--- Fusion: {fusion_type} ---")
        
        if fusion_type == 'cross_attention':
            fusion_model = GNN_BERT_Fusion(
                gnn_hidden_dim=cfg['gnn_hidden_dim'],
                bert_hidden_dim=cfg['bert_hidden_dim'],
                num_tags=num_classes,
                d_k=cfg['d_k']
            ).to(device)
        else:
            fusion_model = EarlyConcatFusion(
                gnn_hidden_dim=cfg['gnn_hidden_dim'],
                bert_hidden_dim=cfg['bert_hidden_dim'],
                num_tags=num_classes
            ).to(device)
        
        optimizer = optim.Adam(fusion_model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        criterion = MultiTaskFusionLoss(alpha=cfg['alpha'], beta=cfg['beta'], multi_label=False)
        
        metrics = {
            "train_loss": [], "val_loss": [],
            "macro_f1": [], "micro_f1": [], "accuracy": [],
            "wall_time_seconds": [],
        }
        
        t_total_start = time.time()
        best_val_f1 = 0.0
        
        for epoch in range(cfg['num_epochs']):
            t_epoch_start = time.time()
            
            # --- Train ---
            fusion_model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            
            for graph_batch, texts in train_loader:
                graph_batch = graph_batch.to(device)
                
                # Get GNN embeddings (frozen)
                with torch.no_grad():
                    _, g = gnn_model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, return_embeddings=True)
                
                # Get BERT embeddings (frozen)
                encoded = tokenizer(
                    list(texts), padding=True, truncation=True,
                    max_length=config['task1_bert']['max_seq_length'],
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                with torch.no_grad():
                    _, _, H_text = bert_model(input_ids, attention_mask, return_embeddings=True)
                
                # Forward through fusion
                optimizer.zero_grad()
                y_hat_tags, _, _ = fusion_model(g, H_text)
                loss = criterion(y_hat_tags, graph_batch.y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            
            avg_train_loss = epoch_loss / max(epoch_batches, 1)
            
            # --- Validate ---
            fusion_model.eval()
            val_loss = 0.0
            val_batches = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for graph_batch, texts in val_loader:
                    graph_batch = graph_batch.to(device)
                    
                    _, g = gnn_model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, return_embeddings=True)
                    
                    encoded = tokenizer(
                        list(texts), padding=True, truncation=True,
                        max_length=config['task1_bert']['max_seq_length'],
                        return_tensors='pt'
                    )
                    input_ids = encoded['input_ids'].to(device)
                    attention_mask = encoded['attention_mask'].to(device)
                    _, _, H_text = bert_model(input_ids, attention_mask, return_embeddings=True)
                    
                    y_hat_tags, _, _ = fusion_model(g, H_text)
                    loss = criterion(y_hat_tags, graph_batch.y)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    preds = torch.argmax(y_hat_tags, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(graph_batch.y.cpu().numpy())
            
            avg_val_loss = val_loss / max(val_batches, 1)
            macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
            acc = accuracy_score(all_labels, all_preds)
            
            epoch_time = time.time() - t_epoch_start
            
            metrics["train_loss"].append(avg_train_loss)
            metrics["val_loss"].append(avg_val_loss)
            metrics["macro_f1"].append(macro_f1)
            metrics["micro_f1"].append(micro_f1)
            metrics["accuracy"].append(acc)
            metrics["wall_time_seconds"].append(epoch_time)
            
            print(f"  Epoch {epoch+1}/{cfg['num_epochs']} -- "
                  f"train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}, "
                  f"macro_f1: {macro_f1:.4f}, acc: {acc:.4f} [{epoch_time:.1f}s]", flush=True)
            
            if macro_f1 > best_val_f1:
                best_val_f1 = macro_f1
                ckpt_name = f'task3_{fusion_type}_best.pt'
                ckpt_path = str(project_root / config['paths']['checkpoints'] / ckpt_name)
                save_checkpoint(fusion_model, optimizer, epoch, metrics, ckpt_path)
                print(f"    -> Saved best checkpoint (macro_f1={macro_f1:.4f})")
        
        total_time = time.time() - t_total_start
        print(f"  Total wall time for {fusion_type}: {total_time:.1f}s ({total_time/60:.1f}min)")
        metrics["total_wall_time"] = total_time
        all_results[fusion_type] = metrics
    
    return all_results


# ============================================================
# TASK 4: InfoNCE Contrastive Dual-Encoder
# ============================================================
def train_contrastive_task4(config, device, gnn_model=None, bert_model=None):
    """Train InfoNCE contrastive dual-encoder on paired (graph, text) data."""
    print("\n" + "="*60)
    print("TASK 4: InfoNCE Contrastive Dual-Encoder")
    print("="*60)
    
    cfg = config['task4_contrastive']
    
    # Load paired data  
    print("Loading paired FMA text+graph data...")
    train_loader, val_loader, test_loader, genre_to_idx = get_fma_text_graph_dataloaders()
    num_classes = len(genre_to_idx)
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    
    # Load or create encoders
    if gnn_model is None:
        gnn_model = GNN_MusicClassifier(
            in_channels=config['task2_gnn']['in_channels'],
            hidden_channels=config['task2_gnn']['hidden_channels'],
            num_classes=num_classes,
            num_layers=config['task2_gnn']['num_layers']
        ).to(device)
        ckpt_path = project_root / config['paths']['checkpoints'] / 'task2_gnn_best.pt'
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), weights_only=False)
            gnn_model.load_state_dict(ckpt['model_state_dict'])
            print("  Loaded pretrained GNN checkpoint")
    
    if bert_model is None:
        bert_model = BERT_MultiLabelClassifier(
            num_tags=num_classes,
            model_name=config['task1_bert']['model_name'],
        ).to(device)
        ckpt_path = project_root / config['paths']['checkpoints'] / 'task1_bert_best.pt'
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), weights_only=False)
            bert_model.load_state_dict(ckpt['model_state_dict'])
            print("  Loaded pretrained BERT checkpoint")
    
    tokenizer = get_tokenizer(config['task1_bert']['model_name'])
    
    # We need a wrapper that provides embedding-level output for the contrastive model
    # GNN encoder: returns graph-level embedding g
    class GNNEmbeddingWrapper(nn.Module):
        def __init__(self, gnn):
            super().__init__()
            self.gnn = gnn
            # Projection head
            self.proj = nn.Linear(config['task2_gnn']['hidden_channels'], cfg['projection_dim'])
        
        def forward(self, x, edge_index, batch):
            _, g = self.gnn(x, edge_index, batch, return_embeddings=True)
            return self.proj(g)
    
    class BERTEmbeddingWrapper(nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert
            # Projection head
            self.proj = nn.Linear(768, cfg['projection_dim'])
        
        def forward(self, input_ids, attention_mask):
            _, cls_output, _ = self.bert(input_ids, attention_mask, return_embeddings=True)
            return self.proj(cls_output)
    
    gnn_wrapper = GNNEmbeddingWrapper(gnn_model).to(device)
    bert_wrapper = BERTEmbeddingWrapper(bert_model).to(device)
    
    contrastive_model = ContrastiveDualEncoder(
        gnn_encoder=gnn_wrapper,
        bert_encoder=bert_wrapper,
        tau=cfg['tau']
    ).to(device)
    
    # Only train projection heads + contrastive params
    optimizer = optim.Adam(
        list(gnn_wrapper.proj.parameters()) + list(bert_wrapper.proj.parameters()),
        lr=cfg['learning_rate'], weight_decay=cfg['weight_decay']
    )
    
    metrics = {
        "train_loss": [], "val_loss": [],
        "wall_time_seconds": [],
        "retrieval_metrics": [],
    }
    
    t_total_start = time.time()
    best_val_loss = float('inf')
    
    for epoch in range(cfg['num_epochs']):
        t_epoch_start = time.time()
        
        # --- Train ---
        contrastive_model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        
        for graph_batch, texts in train_loader:
            graph_batch = graph_batch.to(device)
            
            encoded = tokenizer(
                list(texts), padding=True, truncation=True,
                max_length=config['task1_bert']['max_seq_length'],
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            optimizer.zero_grad()
            loss, S = contrastive_model(graph_batch, input_ids, attention_mask)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
        
        avg_train_loss = epoch_loss / max(epoch_batches, 1)
        
        # --- Validate ---
        contrastive_model.eval()
        val_loss = 0.0
        val_batches = 0
        all_S = []
        
        with torch.no_grad():
            for graph_batch, texts in val_loader:
                graph_batch = graph_batch.to(device)
                
                encoded = tokenizer(
                    list(texts), padding=True, truncation=True,
                    max_length=config['task1_bert']['max_seq_length'],
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                loss, S = contrastive_model(graph_batch, input_ids, attention_mask)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Collect similarity matrices for retrieval eval
                if len(all_S) < 5:  # Only collect a few batches for retrieval
                    all_S.append(S.cpu())
        
        avg_val_loss = val_loss / max(val_batches, 1)
        epoch_time = time.time() - t_epoch_start
        
        # Compute retrieval metrics on first batch
        retrieval = {}
        if len(all_S) > 0:
            retrieval = evaluate_retrieval(all_S[0])
        
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(avg_val_loss)
        metrics["wall_time_seconds"].append(epoch_time)
        metrics["retrieval_metrics"].append(retrieval)
        
        r1_str = ""
        if retrieval:
            r1_str = f", R@1: {retrieval.get('Audio_to_Caption', {}).get('R@1', 0):.3f}"
        
        print(f"  Epoch {epoch+1}/{cfg['num_epochs']} -- "
              f"train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}"
              f"{r1_str} [{epoch_time:.1f}s]", flush=True)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = str(project_root / config['paths']['checkpoints'] / 'task4_contrastive_best.pt')
            save_checkpoint(contrastive_model, optimizer, epoch, metrics, ckpt_path)
            print(f"    -> Saved best checkpoint (val_loss={avg_val_loss:.4f})")
    
    total_time = time.time() - t_total_start
    print(f"  Total wall time: {total_time:.1f}s ({total_time/60:.1f}min)")
    metrics["total_wall_time"] = total_time
    
    return metrics


# ============================================================
# BASELINES
# ============================================================
def train_baselines(config, device):
    """Train baseline models: B1-Random, B2-CNN, B3-BERT-only (frozen)."""
    print("\n" + "="*60)
    print("BASELINES")
    print("="*60)
    
    results = {}
    
    # --- B1: Random Baseline ---
    print("\n--- B1: Random Baseline ---")
    _, val_loader, _, genre_to_idx = get_fma_dataloaders()
    num_classes = len(genre_to_idx)
    
    all_preds = []
    all_labels = []
    for batch in val_loader:
        labels = batch.y.numpy()
        preds = np.random.randint(0, num_classes, size=len(labels))
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    results["B1_random"] = {
        "macro_f1": f1_score(all_labels, all_preds, average='macro', zero_division=0),
        "micro_f1": f1_score(all_labels, all_preds, average='micro', zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
    }
    print(f"  B1 Random — macro_f1: {results['B1_random']['macro_f1']:.4f}, "
          f"micro_f1: {results['B1_random']['micro_f1']:.4f}, "
          f"accuracy: {results['B1_random']['accuracy']:.4f}")
    
    # --- B2: CNN on mean mel-spectrogram features ---
    print("\n--- B2: CNN on mel features ---")
    
    class SimpleCNN(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes),
            )
        
        def forward(self, x):
            return self.fc(x)
    
    # Use graph dataset but only the mean-pooled node features
    train_loader, val_loader, _, genre_to_idx = get_fma_dataloaders()
    num_classes = len(genre_to_idx)
    node_dim = config['graph']['node_feature_dim']
    
    cnn_model = SimpleCNN(node_dim, num_classes).to(device)
    optimizer = optim.Adam(cnn_model.parameters(), lr=config['baselines']['cnn']['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    cnn_metrics = {"train_loss": [], "macro_f1": [], "micro_f1": [], "accuracy": []}
    t_start = time.time()
    
    for epoch in range(config['baselines']['cnn']['num_epochs']):
        cnn_model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            # Mean-pool all node features per graph to get track-level features
            from torch_geometric.nn import global_mean_pool
            track_feats = global_mean_pool(batch.x, batch.batch)
            
            optimizer.zero_grad()
            logits = cnn_model(track_feats)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Validate
        cnn_model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                track_feats = global_mean_pool(batch.x, batch.batch)
                logits = cnn_model(track_feats)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch.y.cpu().numpy())
        
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        
        cnn_metrics["train_loss"].append(epoch_loss / max(n_batches, 1))
        cnn_metrics["macro_f1"].append(macro_f1)
        cnn_metrics["micro_f1"].append(micro_f1)
        cnn_metrics["accuracy"].append(acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{config['baselines']['cnn']['num_epochs']} — "
                  f"loss: {epoch_loss/max(n_batches,1):.4f}, macro_f1: {macro_f1:.4f}, acc: {acc:.4f}")
    
    cnn_metrics["total_wall_time"] = time.time() - t_start
    results["B2_cnn"] = cnn_metrics
    
    # --- B3: BERT-only (frozen BERT + linear head) ---
    print("\n--- B3: BERT-only (frozen encoder) ---")
    
    train_loader, val_loader, _, genre_to_idx = get_fma_text_dataloaders()
    num_classes = len(genre_to_idx)
    
    bert_frozen = BERT_MultiLabelClassifier(
        num_tags=num_classes,
        model_name=config['task1_bert']['model_name'],
        freeze_bert=True  # Frozen BERT
    ).to(device)
    
    tokenizer = get_tokenizer(config['task1_bert']['model_name'])
    optimizer = optim.Adam(bert_frozen.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    bert_metrics = {"train_loss": [], "macro_f1": [], "micro_f1": [], "accuracy": []}
    t_start = time.time()
    
    for epoch in range(5):
        bert_frozen.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_texts, batch_labels, _ in train_loader:
            encoded = tokenizer(
                list(batch_texts), padding=True, truncation=True,
                max_length=config['task1_bert']['max_seq_length'],
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = bert_frozen(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # Validate
        bert_frozen.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_texts, batch_labels, _ in val_loader:
                encoded = tokenizer(
                    list(batch_texts), padding=True, truncation=True,
                    max_length=config['task1_bert']['max_seq_length'],
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                logits = bert_frozen(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_labels.numpy())
        
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        
        bert_metrics["train_loss"].append(epoch_loss / max(n_batches, 1))
        bert_metrics["macro_f1"].append(macro_f1)
        bert_metrics["micro_f1"].append(micro_f1)
        bert_metrics["accuracy"].append(acc)
        
        print(f"  Epoch {epoch+1}/5 — "
              f"loss: {epoch_loss/max(n_batches,1):.4f}, macro_f1: {macro_f1:.4f}, acc: {acc:.4f}")
    
    bert_metrics["total_wall_time"] = time.time() - t_start
    results["B3_bert_frozen"] = bert_metrics
    
    return results


# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GNN-BERT Music Context models")
    parser.add_argument('--task', choices=['1', '2', '3', '4', 'baselines', 'all'], default='all',
                        help='Which task to train')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    all_metrics = {}
    bert_model = None
    gnn_model = None
    
    # Helper to determine which tasks to run
    run_all = args.task == 'all'
    
    if run_all or args.task == '1':
        task1_metrics, bert_model = train_bert_task1(config, device)
        all_metrics["task1_bert"] = task1_metrics
    
    if run_all or args.task == '2':
        task2_metrics, gnn_model = train_gnn_task2(config, device)
        all_metrics["task2_gnn"] = task2_metrics
    
    if run_all or args.task == '3':
        fusion_metrics = train_fusion_task3(config, device, gnn_model=gnn_model, bert_model=bert_model)
        all_metrics["task3_cross_attention"] = fusion_metrics['cross_attention']
        all_metrics["task3_early_concat"] = fusion_metrics['early_concat']
    
    if run_all or args.task == '4':
        task4_metrics = train_contrastive_task4(config, device, gnn_model=gnn_model, bert_model=bert_model)
        all_metrics["task4_contrastive"] = task4_metrics
    
    if run_all or args.task == 'baselines':
        baseline_metrics = train_baselines(config, device)
        all_metrics["baselines"] = baseline_metrics
    
    # Save all metrics
    results_path = str(project_root / config['paths']['results'] / 'metrics.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    # If metrics.json already exists, merge
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r') as f:
                existing = json.load(f)
            existing.update(all_metrics)
            all_metrics = existing
        except (json.JSONDecodeError, Exception):
            pass
    
    with open(results_path, 'w') as f:
        json.dump(all_metrics, f, indent=4, default=str)
    
    print(f"\n{'='*60}")
    print(f"ALL TRAINING COMPLETE")
    print(f"Metrics saved to: {results_path}")
    print(f"{'='*60}")
