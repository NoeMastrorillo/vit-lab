import torch
import torch.nn as nn

import utils


def scaled_dot_product(q, k, v):
    """ Input: q, k: (..., n_patches, key_dim)
               v: (..., n_patches, value_dim)
        Output: (..., n_patches, value_dim)"""
    dk = k.shape[-1]
    A = torch.softmax((q @ k.transpose(-2, -1)) / dk**0.5, dim=-1)
    return A @ v


class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, key_dim, value_dim):
        super().__init__()
        self.q_proj = nn.Linear(embedding_dim, key_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, key_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, value_dim, bias=False)

    def forward(self, x):
        """ Input: (batch_size, n_patches, embedding_dim)
            Output: (batch_size, n_patches, value_dim)
        """
        q = self.q_proj(x)  # (batch_size, n_patches, key_dim)
        k = self.k_proj(x)  # (batch_size, n_patches, key_dim)
        v = self.v_proj(x)  # (batch_size, n_patches, value_dim)
        return scaled_dot_product(q, k, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, key_dim, value_dim, n_heads, dropout=0):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(embedding_dim, key_dim, value_dim) for _ in range(n_heads)])
        self.lin_proj = nn.Linear(n_heads*value_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """ Input: (batch_size, n_patches, embedding_dim)
            Output: same
        """
        x = torch.concat([head(x) for head in self.heads], dim=-1)
        x = self.lin_proj(x)
        x = self.dropout(x)
        return x


class FusedMultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, key_dim, value_dim, n_heads, dropout=0):
        super().__init__()
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.qkv_proj = nn.Linear(embedding_dim, n_heads*(2*key_dim+value_dim), bias=False)
        self.lin_proj = nn.Linear(n_heads*value_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """ Input: (batch_size, n_patches, embedding_dim)
            Output: same
        """
        batch_size, n_patches, _ = x.shape

        qkv = self.qkv_proj(x)  # (batch_size, n_patches, n_heads*key_dim + n_heads*key_dim + n_heads*value_dim)
        q, k, v = qkv.split([self.n_heads*self.key_dim, self.n_heads*self.key_dim, self.n_heads*self.value_dim], dim=-1)
        q = q.view(batch_size, n_patches, self.n_heads, self.key_dim).transpose(1, 2)   # (batch_size, n_heads, n_patches, key_dim)
        k = k.view(batch_size, n_patches, self.n_heads, self.key_dim).transpose(1, 2)
        v = v.view(batch_size, n_patches, self.n_heads, self.value_dim).transpose(1, 2)

        x = scaled_dot_product(q, k, v)     # (batch_size, n_heads, n_patches, value_dim)
        x = x.transpose(1, 2).flatten(2)    # (batch_size, n_patches, n_heads*value_dim)
        x = self.lin_proj(x)
        x = self.dropout(x)
        return x



class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, n_heads, dropout=0, fused=False):
        assert embedding_dim % n_heads == 0

        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        msa_class = MultiHeadSelfAttention
        if fused:
            msa_class = FusedMultiHeadSelfAttention
        self.msa = msa_class(embedding_dim, embedding_dim//n_heads, embedding_dim//n_heads, n_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ Input: (batch_size, n_patches, embedding_dim)
            Output: same
        """
        x = x + self.msa(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size, n_channels, patch_size, embedding_dim, n_layers, mlp_dim, n_heads, dropout=0, fused=False):
        super().__init__()
        self.patch_size = patch_size
        input_dim = n_channels*patch_size*patch_size
        n_patches = (image_size*image_size) // (patch_size*patch_size)

        self.lin_proj = nn.Linear(input_dim, embedding_dim)
        self.cls = nn.Parameter(torch.zeros((1, 1, embedding_dim)))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.pos_embeddings = nn.Parameter(torch.zeros((1, n_patches+1, embedding_dim)))
        self.dropout = nn.Dropout(dropout)
        self.transformer_layers = nn.ModuleList([TransformerLayer(embedding_dim, mlp_dim, n_heads, dropout=dropout, fused=fused) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """ Input: (batch_size, n_channels, height, width)
            Output: (batch_size, n_patches+1, embedding_dim)
        """
        batch_size, n_channels, height, width = x.shape

        # Patching
        x = utils.images_to_patches(x, self.patch_size) # (batch_size, n_patches, input_dim)

        # Embedding
        x = self.lin_proj(x)    # (batch_size, n_patches, embedding_dim)

        # CLS token
        cls = self.cls.expand(batch_size, 1, -1)
        x = torch.cat([cls, x], dim=1)    # (batch_size, n_patches+1, embedding_dim)

        # Positionnal embeddings
        x = x + self.pos_embeddings
        x = self.dropout(x)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        return self.layer_norm(x)


class ViTForClassification(nn.Module):
    def __init__(self, n_classes, image_size, n_channels, patch_size, embedding_dim, n_layers, mlp_dim, n_heads, dropout=0, fused=False):
        super().__init__()
        self.n_classes = n_classes
        self.vit = VisionTransformer(image_size, n_channels, patch_size, embedding_dim, n_layers, mlp_dim, n_heads, dropout=dropout, fused=fused)
        self.linear = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        """ Input: (batch_size, n_channels, height, width)
            Output: (batch_size, n_classes)
        """
        x = self.vit(x)
        cls = x[:, 0]
        return self.linear(cls)

