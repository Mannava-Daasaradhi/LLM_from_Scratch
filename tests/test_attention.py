import torch
import math
import pytest
from model.attention import MultiHeadCausalSelfAttention, scaled_dot_product_attention

def test_output_shape():
    d_model = 512
    n_heads = 8
    attn = MultiHeadCausalSelfAttention(d_model, n_heads)
    
    x = torch.zeros((2, 10, d_model))
    out = attn(x)
    
    assert out.shape == (2, 10, d_model)

def test_causal_mask_prevents_future_attention():
    d_model = 64
    n_heads = 2
    attn = MultiHeadCausalSelfAttention(d_model, n_heads, dropout=0.0)
    
    # Sequence of length 3 (e.g., [A, B, C])
    x = torch.randn(1, 3, d_model)
    
    with torch.no_grad():
        qkv = attn.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(1, 3, n_heads, d_model // n_heads).transpose(1, 2)
        k = k.view(1, 3, n_heads, d_model // n_heads).transpose(1, 2)
        v = v.view(1, 3, n_heads, d_model // n_heads).transpose(1, 2)
        
        # Grab the mask for length 3
        mask = attn.causal_mask[:, :, :3, :3]
        
        # Use our standalone function to get the weights
        _, weights = scaled_dot_product_attention(q, k, v, mask)
        
    # weights shape: (1, n_heads, 3, 3)
    # Position 0 attending to 1 and 2 must be exactly 0.0
    assert weights[0, 0, 0, 1].item() == 0.0
    assert weights[0, 0, 0, 2].item() == 0.0
    
    # Position 1 attending to 2 must be exactly 0.0
    assert weights[0, 0, 1, 2].item() == 0.0

def test_different_heads_different_weights():
    d_model = 64
    n_heads = 4
    attn = MultiHeadCausalSelfAttention(d_model, n_heads, dropout=0.0)
    
    x = torch.randn(1, 5, d_model)
    
    with torch.no_grad():
        qkv = attn.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(1, 5, n_heads, d_model // n_heads).transpose(1, 2)
        k = k.view(1, 5, n_heads, d_model // n_heads).transpose(1, 2)
        v = v.view(1, 5, n_heads, d_model // n_heads).transpose(1, 2)
        
        _, weights = scaled_dot_product_attention(q, k, v)
    
    # Compare head 0 weights with head 1 weights
    # They should initialize differently due to random Linear layer weights
    assert not torch.allclose(weights[0, 0], weights[0, 1])

def test_single_token_works():
    d_model = 64
    n_heads = 2
    attn = MultiHeadCausalSelfAttention(d_model, n_heads)
    
    x = torch.randn(2, 1, d_model)  # Sequence length of 1
    out = attn(x)
    
    assert out.shape == (2, 1, d_model)

def test_gradient_flows():
    d_model = 64
    n_heads = 2
    attn = MultiHeadCausalSelfAttention(d_model, n_heads)
    
    x = torch.randn(2, 5, d_model, requires_grad=True)
    out = attn(x)
    
    # Compute a dummy loss and backward pass
    loss = out.sum()
    loss.backward()
    
    # Assert gradients exist for the parameters
    assert attn.qkv.weight.grad is not None
    assert attn.out_proj.weight.grad is not None

def test_attention_weights_sum_to_one():
    # B, H, T, d_k
    B, H, T, d_k = 2, 4, 10, 16
    q = torch.randn(B, H, T, d_k)
    k = torch.randn(B, H, T, d_k)
    v = torch.randn(B, H, T, d_k)
    
    _, weights = scaled_dot_product_attention(q, k, v)
    
    # Sum over the last dimension (the keys)
    sums = weights.sum(dim=-1)
    
    # All sums should be 1.0 (within 1e-5 tolerance for floating point math)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)