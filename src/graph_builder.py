

import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import Data
import yaml


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        project_root = Path(__file__).parent.parent
        config_path = project_root / config_path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_segment_features(log_mel, chroma, num_segments=10):
    """
    Splits log-mel spectrogram and chroma into temporal segments.
    Returns node feature matrix [num_segments, 140] (128 mel + 12 chroma).
    
    Args:
        log_mel: numpy array [128, T]
        chroma: numpy array [12, T]
        num_segments: Number of segments to divide the track into
    
    Returns:
        numpy array [num_segments, 140]
    """
    num_frames = log_mel.shape[1]
    
    if num_frames < num_segments:
        # If track is shorter than num_segments, pad with zeros
        num_segments = max(1, num_frames)
    
    segment_length = num_frames // num_segments
    
    segment_features = []
    for i in range(num_segments):
        start = i * segment_length
        end = (i + 1) * segment_length if i < num_segments - 1 else num_frames
        
        # Mean-pool mel features over time for this segment
        mel_segment = np.mean(log_mel[:, start:end], axis=1)  # [128]
        
        # Mean-pool chroma features over time for this segment
        chroma_segment = np.mean(chroma[:, start:end], axis=1)  # [12]
        
        # Concatenate: [128 + 12] = [140]
        node_feat = np.concatenate([mel_segment, chroma_segment])
        segment_features.append(node_feat)
    
    return np.array(segment_features, dtype=np.float32)


def build_segment_graph(segment_features, tau=0.8):
    """
    Builds a PyTorch Geometric Data object from segment features.
    
    Edges are added between nodes i and j if:
      (a) |i - j| == 1 (temporal adjacency), OR
      (b) cosine_similarity(node_i, node_j) > tau
    
    Self-loops are excluded.
    
    Args:
        segment_features: numpy array [num_segments, feature_dim]
        tau: Cosine similarity threshold for non-adjacent edges
    
    Returns:
        torch_geometric.data.Data with x, edge_index, edge_attr
    """
    num_nodes = segment_features.shape[0]
    
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(segment_features, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = segment_features / norms
    
    # Pairwise cosine similarity
    sim_matrix = normalized @ normalized.T
    
    edges_src = []
    edges_dst = []
    edge_weights = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            
            is_adjacent = abs(i - j) == 1
            is_similar = sim_matrix[i, j] > tau
            
            if is_adjacent or is_similar:
                edges_src.append(i)
                edges_dst.append(j)
                edge_weights.append(float(sim_matrix[i, j]))
    
    if len(edges_src) > 0:
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty(0, dtype=torch.float32)
    
    x = torch.tensor(segment_features, dtype=torch.float32)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def features_to_graph(feature_dict, num_segments=10, tau=0.8):
    """
    Convert a preprocessed feature dict (from audio_features.py) into a segment graph.
    
    Args:
        feature_dict: dict with 'log_mel' and 'chroma' tensors
        num_segments: Number of segments
        tau: Cosine similarity threshold
    
    Returns:
        torch_geometric.data.Data
    """
    log_mel = feature_dict['log_mel'].numpy()
    chroma = feature_dict['chroma'].numpy()
    
    node_features = extract_segment_features(log_mel, chroma, num_segments)
    graph = build_segment_graph(node_features, tau)
    
    return graph


def process_features_to_graphs(features_dir, graphs_dir, num_segments=10, tau=0.8,
                                skip_existing=True):
    """
    Batch convert all .pt feature files into .pt graph files.
    
    Args:
        features_dir: Directory containing .pt feature files (from audio_features.py)
        graphs_dir: Output directory for .pt graph files
        num_segments: Number of segments per track
        tau: Cosine similarity threshold
        skip_existing: Skip files that already exist in graphs_dir
    
    Returns:
        dict with processing statistics
    """
    os.makedirs(graphs_dir, exist_ok=True)
    
    feature_files = sorted(glob.glob(os.path.join(features_dir, '*.pt')))
    total = len(feature_files)
    
    if total == 0:
        print(f"WARNING: No .pt files found in {features_dir}")
        return {'total': 0, 'processed': 0, 'skipped': 0, 'errors': 0}
    
    print(f"Found {total} feature files in {features_dir}")
    print(f"Graph output: {graphs_dir}")
    print(f"Config: {num_segments} segments, tau={tau}")
    
    processed = 0
    skipped = 0
    errors = 0
    
    t_start = time.time()
    
    for i, feat_path in enumerate(feature_files):
        base_name = os.path.splitext(os.path.basename(feat_path))[0]
        graph_path = os.path.join(graphs_dir, f"{base_name}.pt")
        
        if skip_existing and os.path.exists(graph_path):
            skipped += 1
            continue
        
        try:
            feat_dict = torch.load(feat_path, weights_only=False)
            graph = features_to_graph(feat_dict, num_segments, tau)
            torch.save(graph, graph_path)
            processed += 1
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ERROR [{base_name}]: {e}")
        
        if (i + 1) % 2000 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{total}] {elapsed:.0f}s — "
                  f"{processed} processed, {skipped} skipped, {errors} errors")
    
    total_time = time.time() - t_start
    
    print(f"\n=== Graph Construction Complete ===")
    print(f"  Total:     {total}")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"  Wall time: {total_time:.1f}s")
    
    return {
        'total': total, 'processed': processed,
        'skipped': skipped, 'errors': errors,
        'time_seconds': total_time,
    }


def main():
    parser = argparse.ArgumentParser(description='Build segment graphs from audio features')
    parser.add_argument('--dataset', choices=['fma', 'deam', 'all'], default='all')
    parser.add_argument('--no-skip', action='store_true')
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    project_root = Path(__file__).parent.parent
    
    graph_cfg = config['graph']
    skip_existing = not args.no_skip
    
    if args.dataset in ('fma', 'all'):
        print("\n" + "="*60)
        print("BUILDING FMA SEGMENT GRAPHS")
        print("="*60)
        process_features_to_graphs(
            features_dir=str(project_root / config['paths']['processed_fma']),
            graphs_dir=str(project_root / config['paths']['processed_fma'] / 'graphs'),
            num_segments=graph_cfg['num_segments'],
            tau=graph_cfg['cosine_threshold'],
            skip_existing=skip_existing,
        )
    
    if args.dataset in ('deam', 'all'):
        print("\n" + "="*60)
        print("BUILDING DEAM SEGMENT GRAPHS")
        print("="*60)
        process_features_to_graphs(
            features_dir=str(project_root / config['paths']['processed_deam']),
            graphs_dir=str(project_root / config['paths']['processed_deam'] / 'graphs'),
            num_segments=graph_cfg['num_segments'],
            tau=graph_cfg['cosine_threshold'],
            skip_existing=skip_existing,
        )


if __name__ == '__main__':
    main()
