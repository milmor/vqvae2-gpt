import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import json
import argparse

from vqvae import VQVAE
from transformer_ar import TransformerAR
from transformer_utils import generate, plot_generation


def extract_codes(model, loader, device):
    """Extract codes from dataset using trained VQ-VAE."""
    model.eval()
    all_top, all_bottom = [], []
    with torch.no_grad():
        for img, _ in loader:
            _, _, _, top, bottom = model.encode(img.to(device))
            all_top.append(top.cpu())
            all_bottom.append(bottom.cpu())
    return torch.cat(all_top, 0), torch.cat(all_bottom, 0)


def main(log_interval=100, ckpt_interval=500, gen_interval=500, gen_seed=42, skip_top=False, skip_bottom=False):
    with open('transformer_config.json') as f:
        cfg = json.load(f)
    
    with open('vqvae_config.json') as f:
        vqvae_cfg = json.load(f)
    
    run_name = cfg['run_name']
    ckpt_dir = os.path.join(run_name, 'checkpoint')
    gen_dir = os.path.join(run_name, 'generated')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load VQ-VAE
    vqvae = VQVAE(**vqvae_cfg['model']).to(device)
    vqvae.load_state_dict(torch.load(cfg['vqvae']['checkpoint_path'], map_location=device, weights_only=False)['model'])
    vqvae.eval()
    
    # Extract codes
    transform = transforms.Compose([
        transforms.Resize((vqvae_cfg['dataset']['image_size'], vqvae_cfg['dataset']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('MNIST', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=vqvae_cfg['training']['batch_size'], 
                       shuffle=False, num_workers=vqvae_cfg['training']['num_workers'])
    
    print("Extracting codes...")
    top_codes, bottom_codes = extract_codes(vqvae, loader, device)
    print(f'Extracted codes: top {top_codes.shape}, bottom {bottom_codes.shape}')
    
    criterion = nn.CrossEntropyLoss()
    batch_size = cfg['training']['batch_size']
    
    # Train top
    if not skip_top:
        print("Training top Transformer...")
        model_top = TransformerAR(list(top_codes.shape[1:]), **cfg['model_top']).to(device)
        optimizer_top = torch.optim.Adam(model_top.parameters(), lr=cfg['training']['lr'])
        model_top.train()
        step = 0
        
        while step < cfg['training']['num_steps_top']:
            for i in range(0, len(top_codes), batch_size):
                if step >= cfg['training']['num_steps_top']:
                    break
                batch = top_codes[i:i+batch_size].to(device)
                out, _ = model_top(batch)
                loss = criterion(out, batch)
                optimizer_top.zero_grad()
                loss.backward()
                optimizer_top.step()
                
                if step % log_interval == 0:
                    print(f'Top Step {step}, Loss: {loss.item():.4f}')
                
                if step % ckpt_interval == 0 and step > 0:
                    torch.save(model_top.state_dict(), 
                              os.path.join(ckpt_dir, f'{run_name}_top_{step}.pt'))
                step += 1
        
        torch.save(model_top.state_dict(), os.path.join(ckpt_dir, f'{run_name}_top.pt'))
        model_top.eval()
    else:
        print("Skipping top training, loading from checkpoint...")
        model_top = TransformerAR(list(top_codes.shape[1:]), **cfg['model_top']).to(device)
        model_top.load_state_dict(torch.load(os.path.join(ckpt_dir, f'{run_name}_top.pt'), 
                                            map_location=device, weights_only=False))
        model_top.eval()
    
    # Train bottom
    if not skip_bottom:
        print("Training bottom Transformer...")
        model_bottom = TransformerAR(list(bottom_codes.shape[1:]), **cfg['model_bottom']).to(device)
        optimizer_bottom = torch.optim.Adam(model_bottom.parameters(), lr=cfg['training']['lr'])
        model_bottom.train()
        step = 0
        
        while step < cfg['training']['num_steps_bottom']:
            for i in range(0, len(bottom_codes), batch_size):
                if step >= cfg['training']['num_steps_bottom']:
                    break
                top_batch = top_codes[i:i+batch_size].to(device)
                bottom_batch = bottom_codes[i:i+batch_size].to(device)
                out, _ = model_bottom(bottom_batch, condition=top_batch)
                loss = criterion(out, bottom_batch)
                optimizer_bottom.zero_grad()
                loss.backward()
                optimizer_bottom.step()
                
                if step % log_interval == 0:
                    print(f'Bottom Step {step}, Loss: {loss.item():.4f}')
                
                if step % gen_interval == 0:
                    model_top.eval()
                    model_bottom.eval()
                    gen_images = generate(vqvae, model_top, model_bottom, batch=8, temp=1.0, seed=gen_seed)
                    plot_path = os.path.join(gen_dir, f'generation_step_{step}.png')
                    plot_generation(gen_images, plot_path, step)
                    model_bottom.train()
                
                if step % ckpt_interval == 0 and step > 0:
                    torch.save(model_bottom.state_dict(), 
                              os.path.join(ckpt_dir, f'{run_name}_bottom_{step}.pt'))
                step += 1
        
        torch.save(model_bottom.state_dict(), os.path.join(ckpt_dir, f'{run_name}_bottom.pt'))
    else:
        print("Skipping bottom training.")
    
    print('Training complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--ckpt_interval', type=int, default=500)
    parser.add_argument('--gen_interval', type=int, default=500)
    parser.add_argument('--gen_seed', type=int, default=42, help='Seed for generation during training')
    parser.add_argument('--skip_top', action='store_true', help='Skip top model training')
    parser.add_argument('--skip_bottom', action='store_true', help='Skip bottom model training')
    args = parser.parse_args()
    main(args.log_interval, args.ckpt_interval, args.gen_interval, args.gen_seed, args.skip_top, args.skip_bottom)

