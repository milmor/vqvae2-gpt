import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def generate(model_vqvae, model_top, model_bottom, batch=8, temp=1.0, seed=None):
    """Generate images using autoregressive Transformer models."""
    device = next(model_vqvae.parameters()).device
    dummy = torch.zeros(1, 1, 32, 32, device=device)
    _, _, _, top_codes, bottom_codes = model_vqvae.encode(dummy)
    top_size, bottom_size = list(top_codes.shape[1:]), list(bottom_codes.shape[1:])
    
    # Use fork_rng to isolate seed setting from training randomness
    with torch.random.fork_rng(enabled=seed is not None):
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate top codes autoregressively
        top_sample = torch.zeros(batch, *top_size, dtype=torch.long, device=device)
        cache = {}
        
        # For transformer: generate in raster scan order (flattened sequence)
        # Need to pass full grid but only use logits for current position
        for i in range(top_size[0]):
            for j in range(top_size[1]):
                # Pass the current partial grid (up to current row, all columns in that row)
                partial = top_sample[:, :i+1, :].clone()
                out, cache = model_top(partial, cache=cache)
                prob = torch.softmax(out[:, :, i, j] / temp, 1)
                top_sample[:, i, j] = torch.multinomial(prob, 1).squeeze(-1)
        
        # Generate bottom codes conditioned on top
        bottom_sample = torch.zeros(batch, *bottom_size, dtype=torch.long, device=device)
        cache = {}
        
        # For transformer: generate row by row
        for i in range(bottom_size[0]):
            for j in range(bottom_size[1]):
                partial = bottom_sample[:, :i+1, :].clone()
                out, cache = model_bottom(partial, condition=top_sample, cache=cache)
                prob = torch.softmax(out[:, :, i, j] / temp, 1)
                bottom_sample[:, i, j] = torch.multinomial(prob, 1).squeeze(-1)
    
    return model_vqvae.decode_code(top_sample, bottom_sample).clamp(-1, 1)


def plot_generation(gen_images, save_path, step, temp=1.0, n_samples=8):
    """Plot generated images."""
    gen_images = gen_images[:n_samples].cpu()
    fig, axes = plt.subplots(1, n_samples, figsize=(2*n_samples, 2))
    for i in range(n_samples):
        axes[i].imshow(gen_images[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.suptitle(f'Generated Images - Step {step} (temp={temp})', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

