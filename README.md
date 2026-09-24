GNN-Based BERT for Understanding Context from Music
This repository contains the implementation, models, and deliverables for a multi-modal framework that integrates a Graph Neural Network (GraphSAGE) and a Bidirectional Encoder Representation from Transformers (BERT) for context-aware music representation learning. This project overcomes the limitations of conventional acoustic-only sequence learning by bridging the gap between structural audio graphs and semantic text embeddings.   



System Architecture
The framework represents musical events as a topological structure of sound events and connects them with natural language descriptions, broken down into four primary modeling tasks:   


Task 1: Text Encoder: Uses a 109M parameter bert-base-uncased transformer to encode text for semantic understanding, mapping captions and tags to 16 FMA genres. The token sequences are padded or truncated to a maximum length of 128 tokens.   


Task 2: Graph Encoder: Implements a 3-layer GraphSAGE structure on temporal segment graphs to combine acoustic features from both sequential and non-sequential segments (like repeating motifs) using message passing and a global mean-pooling readout layer.   


Task 3: Cross-Attention Fusion: Adopts a bilateral multi-head attention mechanism to fuse the structural GNN embeddings with the semantic BERT embeddings, optimizing the network using a multi-task loss function.   


Task 4: Contrastive Cross-Modal Alignment: Utilizes a contrastive InfoNCE loss on pairs of graph and text embeddings to project both modalities into a shared 256-dimensional space.   


Datasets and Preprocessing
This pipeline leverages three powerful datasets to train and align the multi-modal network:   


FMA-medium: Used as the source for raw audio clips, providing 16 genre labels and top tags. Tracks are partitioned into 10 temporal regions per track mapped to a graph.   


DEAM: Provides continuous valence and arousal labels used as auxiliary targets for multi-task fusion.   


MusicCaps: Provides expert-curated natural language captions used for contrastive retrieval alignment.   


Audio Preprocessing:

All audio is uniformly resampled to 22,050 Hz using the librosa library, followed by the computation of log-mel spectrograms.   


The generated graphs utilize 140-dimensional node feature vectors, consisting of 128 Mel-spectrogram bands and 12 Chroma features.   


Graph topology is constructed using temporal adjacency edges and cosine similarity edges (threshold set to 0.75).   


Empirical Evaluation
The architecture was evaluated on multi-label tagging, emotion regression, and cross-modal alignment. The Cross-Attention Fusion model significantly outperformed isolated acoustic modeling baselines (such as 2D-CNNs and standalone GraphSAGE).   


Benchmark Results:

Model/Architecture	Macro F1	Accuracy	Wall Time (s)
Random Baseline	0.0449	0.0631	N/A
2D-CNN Baseline	0.1963	0.5216	118.9
BERT-only (Frozen)	0.5690	0.8910	31.8
Task 1: Fine-tuned BERT	1.0000	1.0000	90.6
Task 2: GraphSAGE GNN	0.2893	0.5395	197.9
Task 3: Cross-Attention Fusion	1.0000	1.0000	265.6
Note: All end-to-end forward passes and loss computations were executed using NVIDIA RTX 5090 hardware.   
PDF

Repository Structure & Deliverables
The repository includes the fully implemented model components, basic baselines, and evaluation routines. Key deliverables located in this repository include:   
PDF

results/checkpoints/: Contains the fine-tuned PyTorch model checkpoints.   
results/metrics.json: Contains the raw records of the evaluation metrics.   
graph_samples/: Contains isolated graph samples visualizing the topological structures.   


Latex report..pdf: The comprehensive project document detailing the research, mathematical foundations, and qualitative case studies (including t-SNE visualizations of the learned latent space).   


Author
Mirza Julkawsar Pranto
Department of Computer Science and Engineering, BRAC University
Course: CSE425: Neural Networks   
PDF
