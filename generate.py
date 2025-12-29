import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import json
import argparse

from vqvae import VQVAE
from transformer_ar import TransformerAR
from transformer_utils import generate, plot_generation


def main(batch=8, temp=1.0, output_dir='generated'):
    with open('transformer_config.json') as f:
        cfg = json.load(f)
    
    with open('vqvae_config.json') as f:
        vqvae_cfg = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load VQ-VAE
    vqvae = VQVAE(**vqvae_cfg['model']).to(device)
    vqvae.load_state_dict(torch.load(cfg['vqvae']['checkpoint_path'], 
                                    map_location=device, weights_only=False)['model'])
    vqvae.eval()
    
    # Load models
    run_name = cfg['run_name']
    ckpt_dir = os.path.join(run_name, 'checkpoint')
    
    # Get code shapes
    transform = transforms.Compose([
        transforms.Resize((vqvae_cfg['dataset']['image_size'], vqvae_cfg['dataset']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('MNIST', train=True, download=True, transform=transform)
    dummy_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    dummy_img, _ = next(iter(dummy_loader))
    _, _, _, top_codes, bottom_codes = vqvae.encode(dummy_img.to(device))
    
    model_top = TransformerAR(list(top_codes.shape[1:]), **cfg['model_top']).to(device)
    model_bottom = TransformerAR(list(bottom_codes.shape[1:]), **cfg['model_bottom']).to(device)
    
    model_top.load_state_dict(torch.load(os.path.join(ckpt_dir, f'{run_name}_top.pt'), 
                                         map_location=device, weights_only=False))
    model_top.eval()
    
    model_bottom.load_state_dict(torch.load(os.path.join(ckpt_dir, f'{run_name}_bottom.pt'), 
                                           map_location=device, weights_only=False))
    model_bottom.eval()
    
    # Generate images
    print(f'Generating {batch} images with temperature {temp}...')
    gen_images = generate(vqvae, model_top, model_bottom, batch=batch, temp=temp)
    
    # Save images
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'generated_temp_{temp}.png')
    plot_generation(gen_images, plot_path, step=0, temp=temp, n_samples=batch)
    print(f'Generated images saved to {plot_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='Number of images to generate')
    parser.add_argument('--temp', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--output_dir', type=str, default='generated', help='Output directory')
    args = parser.parse_args()
    main(args.batch, args.temp, args.output_dir)

