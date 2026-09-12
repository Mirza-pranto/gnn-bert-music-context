

import os
import glob
import shutil
import random

def isolate_graph_samples(processed_dir, output_dir, num_samples=20, seed=42):

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    pt_files = glob.glob(os.path.join(processed_dir, '**', '*.pt'), recursive=True)

    if len(pt_files) == 0:
        print(f"No .pt files found in {processed_dir}.")
        print("Generating synthetic graph samples for submission demo...")
        generate_synthetic_samples(output_dir, num_samples)
        return

    selected = random.sample(pt_files, min(num_samples, len(pt_files)))

    for src in selected:
        dst = os.path.join(output_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"  Copied {os.path.basename(src)}")

    print(f"\nDone. {len(selected)} graph samples saved to {output_dir}")


def generate_synthetic_samples(output_dir, num_samples=20):

    import torch
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        # Simulate a 128-bin log-mel spectrogram with ~500 frames
        num_frames = random.randint(400, 600)
        spec = torch.tensor(np.random.randn(128, num_frames).astype(np.float32))
        path = os.path.join(output_dir, f"sample_{i:03d}.pt")
        torch.save(spec, path)
        print(f"  Generated {os.path.basename(path)}  (128 x {num_frames})")

    print(f"\nDone. {num_samples} synthetic graph samples saved to {output_dir}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    output_dir = os.path.join(project_root, 'graph_samples')

    isolate_graph_samples(processed_dir, output_dir, num_samples=20)
