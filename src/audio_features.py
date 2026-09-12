

import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
import librosa
import yaml
from pathlib import Path


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    # Try relative to project root
    if not os.path.isabs(config_path):
        project_root = Path(__file__).parent.parent
        config_path = project_root / config_path
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_features(audio_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extract log-mel spectrogram and chroma features from a single audio file.
    
    Args:
        audio_path: Path to audio file (mp3/wav)
        sr: Target sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
    
    Returns:
        dict with 'log_mel' (Tensor [n_mels, T]), 'chroma' (Tensor [12, T]),
        'sr' (int), 'duration' (float)
    
    Raises:
        RuntimeError: If audio cannot be loaded or features cannot be extracted
    """
    # Load and resample audio
    y, loaded_sr = librosa.load(audio_path, sr=sr)
    
    if len(y) == 0:
        raise RuntimeError(f"Empty audio file: {audio_path}")
    
    duration = len(y) / sr
    
    # Extract 128-bin log-mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Extract 12-bin chroma features
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # Verify shapes are compatible (same number of time frames)
    min_frames = min(log_mel.shape[1], chroma.shape[1])
    log_mel = log_mel[:, :min_frames]
    chroma = chroma[:, :min_frames]
    
    return {
        'log_mel': torch.tensor(log_mel, dtype=torch.float32),
        'chroma': torch.tensor(chroma, dtype=torch.float32),
        'sr': sr,
        'duration': duration,
    }


def process_dataset(audio_dir, output_dir, sr=22050, n_mels=128, n_fft=2048,
                    hop_length=512, file_ext='*.mp3', skip_existing=True):
    """
    Batch-process all audio files in a directory, saving features as .pt files.
    
    Args:
        audio_dir: Directory containing audio files (searched recursively)
        output_dir: Directory to save .pt feature files
        sr, n_mels, n_fft, hop_length: Audio feature parameters
        file_ext: Glob pattern for audio files
        skip_existing: If True, skip files that already have a .pt output
    
    Returns:
        dict with processing statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio files
    audio_files = sorted(glob.glob(
        os.path.join(audio_dir, '**', file_ext), recursive=True
    ))
    
    total = len(audio_files)
    if total == 0:
        print(f"WARNING: No {file_ext} files found in {audio_dir}")
        return {'total': 0, 'processed': 0, 'skipped': 0, 'errors': 0}
    
    print(f"Found {total} audio files in {audio_dir}")
    print(f"Output directory: {output_dir}")
    
    processed = 0
    skipped = 0
    errors = 0
    error_files = []
    
    t_start = time.time()
    
    for i, audio_path in enumerate(audio_files):
        # Derive output filename from the audio filename (without subdirs)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.pt")
        
        # Skip if already processed
        if skip_existing and os.path.exists(output_path):
            skipped += 1
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{total}] {elapsed:.0f}s elapsed — "
                      f"{processed} processed, {skipped} skipped, {errors} errors")
            continue
        
        try:
            features = extract_features(
                audio_path, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
            )
            torch.save(features, output_path)
            processed += 1
        except Exception as e:
            errors += 1
            error_files.append((audio_path, str(e)))
            if errors <= 10:
                print(f"  ERROR [{base_name}]: {e}")
        
        # Progress report every 500 files
        if (i + 1) % 500 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (processed + skipped) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{total}] {elapsed:.0f}s elapsed, "
                  f"~{eta:.0f}s remaining — "
                  f"{processed} processed, {skipped} skipped, {errors} errors")
    
    total_time = time.time() - t_start
    
    stats = {
        'total': total,
        'processed': processed,
        'skipped': skipped,
        'errors': errors,
        'time_seconds': total_time,
    }
    
    print(f"\n=== Processing Complete ===")
    print(f"  Total files:  {total}")
    print(f"  Processed:    {processed}")
    print(f"  Skipped:      {skipped}")
    print(f"  Errors:       {errors}")
    print(f"  Wall time:    {total_time:.1f}s ({total_time/60:.1f} min)")
    
    if errors > 0 and errors <= 20:
        print(f"\n  Error files:")
        for path, err in error_files[:20]:
            print(f"    {os.path.basename(path)}: {err}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Extract audio features for GNN-BERT project')
    parser.add_argument('--dataset', choices=['fma', 'deam', 'all'], default='all',
                        help='Which dataset to process')
    parser.add_argument('--no-skip', action='store_true',
                        help='Re-process files even if output exists')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    project_root = Path(__file__).parent.parent
    
    audio_cfg = config['audio']
    skip_existing = not args.no_skip
    
    if args.dataset in ('fma', 'all'):
        print("\n" + "="*60)
        print("PROCESSING FMA-MEDIUM AUDIO")
        print("="*60)
        process_dataset(
            audio_dir=str(project_root / config['paths']['fma_audio']),
            output_dir=str(project_root / config['paths']['processed_fma']),
            sr=audio_cfg['sample_rate'],
            n_mels=audio_cfg['n_mels'],
            n_fft=audio_cfg['n_fft'],
            hop_length=audio_cfg['hop_length'],
            file_ext='*.mp3',
            skip_existing=skip_existing,
        )
    
    if args.dataset in ('deam', 'all'):
        print("\n" + "="*60)
        print("PROCESSING DEAM AUDIO")
        print("="*60)
        process_dataset(
            audio_dir=str(project_root / config['paths']['deam_audio']),
            output_dir=str(project_root / config['paths']['processed_deam']),
            sr=audio_cfg['sample_rate'],
            n_mels=audio_cfg['n_mels'],
            n_fft=audio_cfg['n_fft'],
            hop_length=audio_cfg['hop_length'],
            file_ext='*.mp3',
            skip_existing=skip_existing,
        )


if __name__ == '__main__':
    main()
