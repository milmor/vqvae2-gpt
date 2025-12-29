import torch
import matplotlib.pyplot as plt
import os


def plot_reconstruction(original, reconstructed, save_path, step, n_samples=8):
    """Plot original vs reconstructed images side by side."""
    original = original[:n_samples].cpu()
    reconstructed = reconstructed[:n_samples].cpu()
    
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    for j in range(n_samples):
        axes[0, j].imshow(original[j].squeeze(), cmap='gray')
        axes[1, j].imshow(reconstructed[j].squeeze(), cmap='gray')
        axes[0, j].axis('off')
        axes[1, j].axis('off')
    
    axes[0, 0].set_ylabel('Original', fontsize=10)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=10)
    plt.suptitle(f'Step {step}', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

