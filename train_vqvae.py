import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import json
import argparse

from vqvae import VQVAE
from vqvae_utils import plot_reconstruction


def main(log_interval=10, ckpt_interval=200, plot_interval=100):
    with open('vqvae_config.json') as f:
        cfg = json.load(f)
    
    run_name = cfg['run_name']
    # Create run directories
    ckpt_dir = os.path.join(run_name, 'checkpoint')
    log_dir = os.path.join(run_name, 'log')
    plot_dir = os.path.join(run_name, 'plots')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize((cfg['dataset']['image_size'], cfg['dataset']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('MNIST', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], 
                       shuffle=True, num_workers=cfg['training']['num_workers'])
    
    # Model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VQVAE(**cfg['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    
    # Training
    model.train()
    criterion = nn.MSELoss()
    step = 0
    
    while step < cfg['training']['num_steps']:
        for img, _ in loader:
            if step >= cfg['training']['num_steps']:
                break
                
            img = img.to(device)
            out, latent_loss = model(img)
            loss = criterion(out, img) + cfg['training']['latent_loss_weight'] * latent_loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % log_interval == 0:
                print(f'Step {step}, Loss: {loss.item():.4f}')
            
            if step % plot_interval == 0:
                model.eval()
                with torch.no_grad():
                    recon, _ = model(img[:8])
                    plot_path = os.path.join(plot_dir, f'reconstruction_step_{step}.png')
                    plot_reconstruction(img[:8], recon, plot_path, step)
                model.train()
            
            if step % ckpt_interval == 0 and step > 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(ckpt_dir, f'{run_name}_{step}.pt'))
            
            step += 1
    
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(ckpt_dir, f'{run_name}.pt'))
    print('Training complete! Checkpoint saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--ckpt_interval', type=int, default=200)
    parser.add_argument('--plot_interval', type=int, default=200)
    args = parser.parse_args()
    main(args.log_interval, args.ckpt_interval, args.plot_interval)

