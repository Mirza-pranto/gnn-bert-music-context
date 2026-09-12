import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader as PyGDataLoader
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys

def load_config(config_path='config.yaml'):
    if not os.path.isabs(config_path):
        project_root = Path(__file__).parent.parent
        config_path = project_root / config_path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class FMAGraphDataset(Dataset):
    def __init__(self, split='training', config_path='config.yaml'):
        super().__init__()
        self.config = load_config(config_path)
        self.project_root = Path(__file__).parent.parent
        self.graphs_dir = self.project_root / self.config['paths']['processed_fma'] / 'graphs'
        
        # Load FMA metadata
        fma_raw_dir = self.project_root / self.config['paths']['fma_raw']
        sys.path.insert(0, str(fma_raw_dir))
        import utils as fma_utils
        
        tracks_path = fma_raw_dir / 'fma_metadata' / 'tracks.csv'
        tracks = fma_utils.load(str(tracks_path))
        
        # Filter to medium subset
        medium = tracks[tracks[('set', 'subset')] <= 'medium']
        
        # Filter by split
        split_data = medium[medium[('set', 'split')] == split]
        
        # Valid genres mapping
        genres = split_data[('track', 'genre_top')].dropna().unique()
        self.genre_to_idx = {g: i for i, g in enumerate(sorted(genres))}
        
        self.samples = []
        for track_id, row in split_data.iterrows():
            genre = row[('track', 'genre_top')]
            if pd.isna(genre):
                continue
            
            # Formatting track_id to match filename, e.g., 000002.pt
            track_id_str = f"{track_id:06d}"
            graph_path = self.graphs_dir / f"{track_id_str}.pt"
            
            if graph_path.exists():
                self.samples.append({
                    'path': str(graph_path),
                    'label': self.genre_to_idx[genre],
                    'track_id': track_id
                })
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        graph = torch.load(sample['path'], weights_only=False)
        graph.y = torch.tensor(sample['label'], dtype=torch.long)
        return graph

def get_fma_dataloaders(config_path='config.yaml'):
    config = load_config(config_path)
    batch_size = config['task2_gnn']['batch_size']
    
    train_ds = FMAGraphDataset('training', config_path)
    val_ds = FMAGraphDataset('validation', config_path)
    test_ds = FMAGraphDataset('test', config_path)
    
    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_ds.genre_to_idx

class FMATextDataset(Dataset):
    def __init__(self, split='training', config_path='config.yaml'):
        super().__init__()
        self.config = load_config(config_path)
        self.project_root = Path(__file__).parent.parent
        
        fma_raw_dir = self.project_root / self.config['paths']['fma_raw']
        sys.path.insert(0, str(fma_raw_dir))
        import utils as fma_utils
        
        tracks_path = fma_raw_dir / 'fma_metadata' / 'tracks.csv'
        tracks = fma_utils.load(str(tracks_path))
        medium = tracks[tracks[('set', 'subset')] <= 'medium']
        split_data = medium[medium[('set', 'split')] == split]
        
        genres = split_data[('track', 'genre_top')].dropna().unique()
        self.genre_to_idx = {g: i for i, g in enumerate(sorted(genres))}
        
        self.samples = []
        for track_id, row in split_data.iterrows():
            genre = row[('track', 'genre_top')]
            if pd.isna(genre):
                continue
            
            # create pseudo-caption from metadata
            artist = row[('artist', 'name')]
            title = row[('track', 'title')]
            artist_str = str(artist) if pd.notna(artist) else "Unknown Artist"
            title_str = str(title) if pd.notna(title) else "Unknown Title"
            caption = f"A {genre} song titled {title_str} by {artist_str}."
            
            self.samples.append({
                'text': caption,
                'label': self.genre_to_idx[genre],
                'track_id': track_id
            })
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['text'], torch.tensor(sample['label'], dtype=torch.long), sample['track_id']

def get_fma_text_dataloaders(config_path='config.yaml'):
    config = load_config(config_path)
    batch_size = config['task1_bert']['batch_size']
    
    train_ds = FMATextDataset('training', config_path)
    val_ds = FMATextDataset('validation', config_path)
    test_ds = FMATextDataset('test', config_path)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_ds.genre_to_idx

class DEAMGraphDataset(Dataset):
    def __init__(self, split='training', config_path='config.yaml'):
        super().__init__()
        self.config = load_config(config_path)
        self.project_root = Path(__file__).parent.parent
        self.graphs_dir = self.project_root / self.config['paths']['processed_deam'] / 'graphs'
        
        deam_ann_path = self.project_root / self.config['paths']['deam_annotations'] / 'static_annotations_averaged_songs_1_2000.csv'
        deam_ann_path2 = self.project_root / self.config['paths']['deam_annotations'] / 'static_annotations_averaged_songs_2000_2058.csv'
        
        df1 = pd.read_csv(deam_ann_path)
        df2 = pd.read_csv(deam_ann_path2)
        df = pd.concat([df1, df2])
        
        # simple split: first 80% train, next 10% val, next 10% test
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        n = len(df)
        if split == 'training':
            df = df.iloc[:int(0.8*n)]
        elif split == 'validation':
            df = df.iloc[int(0.8*n):int(0.9*n)]
        else:
            df = df.iloc[int(0.9*n):]
            
        self.samples = []
        for _, row in df.iterrows():
            track_id = int(row['song_id'])
            # The filenames for DEAM are like '10.pt', '1000.pt' (no leading zeros)
            graph_path = self.graphs_dir / f"{track_id}.pt"
            if graph_path.exists():
                self.samples.append({
                    'path': str(graph_path),
                    'valence': row[' valence_mean'],
                    'arousal': row[' arousal_mean'],
                    'track_id': track_id
                })
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        graph = torch.load(sample['path'], weights_only=False)
        # add valence and arousal to graph object
        graph.valence = torch.tensor(sample['valence'], dtype=torch.float32)
        graph.arousal = torch.tensor(sample['arousal'], dtype=torch.float32)
        return graph

def get_deam_dataloaders(config_path='config.yaml'):
    config = load_config(config_path)
    batch_size = config['task2_gnn']['batch_size']
    
    train_ds = DEAMGraphDataset('training', config_path)
    val_ds = DEAMGraphDataset('validation', config_path)
    test_ds = DEAMGraphDataset('test', config_path)
    
    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

class FMATextGraphDataset(Dataset):
    def __init__(self, split='training', config_path='config.yaml'):
        super().__init__()
        self.config = load_config(config_path)
        self.project_root = Path(__file__).parent.parent
        self.graphs_dir = self.project_root / self.config['paths']['processed_fma'] / 'graphs'
        
        fma_raw_dir = self.project_root / self.config['paths']['fma_raw']
        sys.path.insert(0, str(fma_raw_dir))
        import utils as fma_utils
        
        tracks_path = fma_raw_dir / 'fma_metadata' / 'tracks.csv'
        tracks = fma_utils.load(str(tracks_path))
        medium = tracks[tracks[('set', 'subset')] <= 'medium']
        split_data = medium[medium[('set', 'split')] == split]
        
        genres = split_data[('track', 'genre_top')].dropna().unique()
        self.genre_to_idx = {g: i for i, g in enumerate(sorted(genres))}
        
        self.samples = []
        for track_id, row in split_data.iterrows():
            genre = row[('track', 'genre_top')]
            if pd.isna(genre):
                continue
                
            track_id_str = f"{track_id:06d}"
            graph_path = self.graphs_dir / f"{track_id_str}.pt"
            
            if not graph_path.exists():
                continue
            
            artist = row[('artist', 'name')]
            title = row[('track', 'title')]
            artist_str = str(artist) if pd.notna(artist) else "Unknown Artist"
            title_str = str(title) if pd.notna(title) else "Unknown Title"
            caption = f"A {genre} song titled {title_str} by {artist_str}."
            
            self.samples.append({
                'path': str(graph_path),
                'text': caption,
                'label': self.genre_to_idx[genre],
                'track_id': track_id
            })
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        graph = torch.load(sample['path'], weights_only=False)
        graph.y = torch.tensor(sample['label'], dtype=torch.long)
        return graph, sample['text']

def get_fma_text_graph_dataloaders(config_path='config.yaml'):
    config = load_config(config_path)
    batch_size = config['task3_fusion']['batch_size']
    
    train_ds = FMATextGraphDataset('training', config_path)
    val_ds = FMATextGraphDataset('validation', config_path)
    test_ds = FMATextGraphDataset('test', config_path)
    
    # Custom collate function to handle (graph, text) pairs
    def collate_fn(batch):
        graphs = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        
        from torch_geometric.data import Batch
        graph_batch = Batch.from_data_list(graphs)
        
        return graph_batch, texts

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, train_ds.genre_to_idx
