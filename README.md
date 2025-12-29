# VQ-VAE-2 GPT

Hierarchical VQ-VAE-2 with GPT-style transformer for autoregressive image generation.

## About

This repository implements a two-stage generative model combining **VQ-VAE-2** (Vector Quantized Variational Autoencoder) with **autoregressive Transformers** for image generation. This architecture follows the same design principles used in generative models:

- **DALL-E** (OpenAI): Uses discrete VAE + Transformer to generate images from text
- **Jukebox** (OpenAI): Uses hierarchical VQ-VAE + Transformer for music generation

The key idea is to compress images into discrete tokens using VQ-VAE, then model the distribution of these tokens using autoregressive Transformers (similar to GPT for text). This approach enables high-quality, diverse image generation.

## Quick Start

### 1. Train VQ-VAE
```bash
python train_vqvae.py
```

### 2. Train Autoregressive Model
```bash
python train_transformer.py
```

### 3. Generate Images
```bash
python generate.py --batch 8 --temp 1.0
```

## Configuration

- `vqvae_config.json` - VQ-VAE model and training hyperparameters
- `transformer_config.json` - Transformer model config

## Features

- **VQ-VAE-2**: Hierarchical vector quantization with top and bottom codes
- **Transformer**: ImageGPT-style autoregressive generation
- **Two-stage generation**: Top codes → Bottom codes (conditioned on top)

## Files

- `vqvae.py` - VQ-VAE-2 implementation
- `transformer_ar.py` - GPT-style transformer
- `train_vqvae.py` - VQ-VAE training
- `train_transformer.py` - Transformer training
- `generate.py` - Image generation
- `transformer_utils.py` - Generation utilities

## Attribution

- This implementation uses code from the [rosinality/vq-vae-2-pytorch](https://github.com/rosinality/vq-vae-2-pytorch) repository, licensed under MIT and Apache Version 2.0 licenses.

## Licence

MIT
