import torch
import math
import pytest
from model.embedding import TokenEmbedding, SinusoidalPositionalEncoding, LearnedPositionalEncoding

def test_token_embedding_shape():
    vocab_size = 1000
    d_model = 512
    batch_size = 2
    seq_len = 10
    
    emb = TokenEmbedding(vocab_size, d_model)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    out = emb(x)
    
    # Assert output shape is (batch_size, seq_len, d_model)
    assert out.shape == (batch_size, seq_len, d_model)

def test_sinusoidal_pe_shape():
    d_model = 512
    max_seq_len = 5000
    batch_size = 2
    seq_len = 10
    
    pe = SinusoidalPositionalEncoding(d_model, max_seq_len)
    
    # Create a dummy tensor simulating token embeddings
    x = torch.zeros((batch_size, seq_len, d_model))
    out = pe(x)
    
    # Positional encoding shouldn't alter the shape
    assert out.shape == (batch_size, seq_len, d_model)

def test_sinusoidal_pe_is_not_learned():
    d_model = 512
    pe = SinusoidalPositionalEncoding(d_model)
    
    # Check that there are no trainable parameters
    trainable_params = [p for p in pe.parameters() if p.requires_grad]
    assert len(trainable_params) == 0, "Sinusoidal PE should not have trainable parameters"
    
    # Verify the PE matrix was correctly registered as a buffer
    assert hasattr(pe, 'pe'), "PE buffer is missing"
    assert not pe.pe.requires_grad, "PE buffer should not require gradients"

def test_pe_values_sin_cos():
    d_model = 512
    pe_layer = SinusoidalPositionalEncoding(d_model)
    
    # pe buffer shape is (1, max_seq_len, d_model)
    # Extract position 0, dimension 0 (sin) and dimension 1 (cos)
    val_sin = pe_layer.pe[0, 0, 0].item()
    val_cos = pe_layer.pe[0, 0, 1].item()
    
    # Check that they match sin(0) and cos(0)
    assert math.isclose(val_sin, 0.0, abs_tol=1e-5), f"Expected 0.0, got {val_sin}"
    assert math.isclose(val_cos, 1.0, abs_tol=1e-5), f"Expected 1.0, got {val_cos}"

def test_learned_pe_is_learned():
    d_model = 512
    max_seq_len = 5000
    pe = LearnedPositionalEncoding(d_model, max_seq_len)
    
    # Check that the module contains trainable parameters
    trainable_params = list(pe.parameters())
    assert len(trainable_params) > 0, "Learned PE is missing trainable parameters"
    
    # Specifically assert the embedding weights require gradients
    assert pe.embedding.weight.requires_grad is True, "Embedding weights should require gradients"