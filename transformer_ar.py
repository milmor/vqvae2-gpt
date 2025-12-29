import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class CausalSelfAttention(nn.Module):
    """Causal self-attention."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, cache=None):
        B, N, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Causal mask
        mask = torch.tril(torch.ones(N, N, device=x.device))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Cross-attention for conditioning."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, condition, cache=None):
        B, N, D = x.shape
        B_cond, M, D_cond = condition.shape
        
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        if cache is not None and 'cond_kv' in cache:
            k, v = cache['cond_kv']
        else:
            kv = self.to_kv(condition).chunk(2, dim=-1)
            k, v = kv[0], kv[1]
            k = k.view(B_cond, M, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B_cond, M, self.num_heads, self.head_dim).transpose(1, 2)
            if cache is not None:
                cache['cond_kv'] = (k, v)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1, has_cross_attn=False):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, dropout)
        self.cross_attn = CrossAttention(dim, num_heads, dropout) if has_cross_attn else None
        self.ff = FeedForward(dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim) if has_cross_attn else None
    
    def forward(self, x, condition=None, cache=None):
        x = x + self.attn(self.norm1(x), cache)
        if condition is not None and self.cross_attn is not None:
            if cache is None:
                cache = {}
            cache_key = 'cross_attn'
            if cache_key not in cache:
                cache[cache_key] = {}
            x = x + self.cross_attn(self.norm2(x), condition, cache[cache_key])
            x = x + self.ff(self.norm3(x))
        else:
            x = x + self.ff(self.norm2(x))
        return x


class TransformerAR(nn.Module):
    """Autoregressive Transformer for discrete code generation (ImageGPT style)."""
    def __init__(self, shape, n_class, dim=512, num_heads=8, depth=6, ff_dim=None, dropout=0.1):
        super().__init__()
        self.shape = shape
        self.n_class = n_class
        self.dim = dim
        self.seq_len = shape[0] * shape[1]
        
        if ff_dim is None:
            ff_dim = dim * 4
        
        # Start of sequence token (like ImageGPT)
        self.sos = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.sos, std=0.02)
        
        self.embedding = nn.Embedding(n_class, dim)
        # Learned positional embeddings for sequence positions (raster scan order)
        self.pos_embedding = nn.Embedding(self.seq_len, dim)
        
        # All blocks support cross-attention (will be None if condition not provided)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, ff_dim, dropout, has_cross_attn=True) 
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, n_class, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.to_logits.weight, std=0.02)
        
    def forward(self, input, condition=None, cache=None):
        """
        Args:
            input: (B, H, W) - integer codes (can be partial during generation)
            condition: Optional (B, H_cond, W_cond) - conditioning codes
            cache: Optional dict for caching
        
        Returns:
            logits: (B, n_class, H, W) - logits for next token prediction
            cache: Updated cache dict
        """
        B, H, W = input.shape
        input_flat = input.view(B, -1)  # (B, seq_len) - raster scan order
        seq_len = input_flat.size(1)
        
        # Embed tokens
        x = self.embedding(input_flat)  # (B, seq_len, dim)
        
        # ImageGPT style: prepend SOS token and shift input
        # SOS token for first position, then tokens shifted by one
        sos_expanded = self.sos.unsqueeze(0).unsqueeze(0).expand(B, 1, self.dim)  # (B, 1, dim)
        x_shifted = torch.cat([sos_expanded, x[:, :-1, :]], dim=1)  # (B, seq_len, dim)
        
        # Add positional embeddings (for sequence positions in raster scan order)
        positions = torch.arange(seq_len, device=input.device).unsqueeze(0).expand(B, -1)  # (B, seq_len)
        pos_emb = self.pos_embedding(positions)  # (B, seq_len, dim)
        x = x_shifted + pos_emb
        
        # Process condition if provided
        cond_emb = None
        if condition is not None:
            cond_flat = condition.view(B, -1)  # (B, H_cond*W_cond)
            cond_emb = self.embedding(cond_flat)  # (B, H_cond*W_cond, dim)
        
        # Apply transformer blocks with proper caching
        if cache is None:
            cache = {}
        
        for i, block in enumerate(self.blocks):
            cache_key = f'block_{i}'
            if cache_key not in cache:
                cache[cache_key] = {}
            x = block(x, condition=cond_emb, cache=cache[cache_key])
        
        x = self.norm(x)
        logits = self.to_logits(x)  # (B, seq_len, n_class)
        
        # Reshape to (B, n_class, H, W) - map back to 2D grid
        if seq_len == self.seq_len:
            logits = logits.view(B, self.shape[0], self.shape[1], self.n_class).permute(0, 3, 1, 2)
        else:
            # Partial sequence during generation - pad to full size
            full_logits = torch.zeros(B, self.seq_len, self.n_class, device=logits.device, dtype=logits.dtype)
            full_logits[:, :seq_len, :] = logits
            logits = full_logits.view(B, self.shape[0], self.shape[1], self.n_class).permute(0, 3, 1, 2)
        
        return logits, cache

