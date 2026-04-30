import torch
import torch.optim as optim
import pytest
from dataclasses import dataclass
from model.transformer import GPT

# Dummy config for testing
@dataclass
class ModelConfig:
    vocab_size: int = 1000
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    max_seq_len: int = 128
    dropout: float = 0.0

@pytest.fixture
def dummy_config():
    return ModelConfig()

@pytest.fixture
def model(dummy_config):
    return GPT(dummy_config)

def test_forward_shape(model, dummy_config):
    batch_size = 2
    seq_len = 32
    x = torch.randint(0, dummy_config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(x)
    
    assert logits.shape == (batch_size, seq_len, dummy_config.vocab_size)
    assert loss is None

def test_forward_with_targets_returns_loss(model, dummy_config):
    batch_size = 2
    seq_len = 32
    x = torch.randint(0, dummy_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, dummy_config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(x, targets=targets)
    
    assert logits.shape == (batch_size, seq_len, dummy_config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # Assert it's a scalar tensor

def test_loss_decreases_on_single_batch(model, dummy_config):
    # This is a classic ML sanity check: can the model overfit a single batch?
    batch_size = 2
    seq_len = 32
    x = torch.randint(0, dummy_config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, dummy_config.vocab_size, (batch_size, seq_len))
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    losses = []
    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        _, loss = model(x, targets=targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    # Assert the loss after 10 steps is strictly less than the initial loss
    assert losses[-1] < losses[0]

def test_generate_returns_longer_sequence(model, dummy_config):
    input_ids = torch.tensor([[2, 100, 101]]) # Start with 3 tokens (e.g., BOS + 2 words)
    max_new = 20
    
    output_ids = model.generate(input_ids, max_new_tokens=max_new)
    
    # Check that it generated exactly max_new tokens (unless it hit EOS early, 
    # but with random weights, hitting EOS early is extremely unlikely).
    assert output_ids.shape[1] > input_ids.shape[1]
    assert output_ids.shape[1] <= input_ids.shape[1] + max_new

def test_generate_respects_temperature(model, dummy_config):
    # Fix the seed so the only variance is temperature
    torch.manual_seed(42)
    input_ids = torch.tensor([[2, 100]])
    
    # T -> 0 should act greedy (highly deterministic)
    out_cold = model.generate(input_ids, max_new_tokens=10, temperature=0.01)
    
    # Restore seed and run with high temperature (highly random)
    torch.manual_seed(42)
    out_hot = model.generate(input_ids, max_new_tokens=10, temperature=2.0)
    
    # The sequences should diverge
    assert not torch.equal(out_cold, out_hot)

def test_parameter_count(model, dummy_config):
    # Manual calculation based on the architecture:
    # 1. Embeddings: vocab_size * d_model = 1000 * 64 = 64,000
    # 2. Sinusoidal PE: 0 learned params
    # 3. Transformer Blocks (x2):
    #    - LayerNorms (x2): (64 weight + 64 bias) * 2 = 256
    #    - Attention: QKV (64 * 192), Out (64 * 64) = 12,288 + 4,096 = 16,384
    #    - FFN: Linear1 (64 * 256 + 256 bias), Linear2 (256 * 64 + 64 bias) = 16,640 + 16,448 = 33,088
    #    Block Total: 256 + 16,384 + 33,088 = 49,728
    #    For 2 blocks: 49,728 * 2 = 99,456
    # 4. Final LayerNorm: 64 weight + 64 bias = 128
    # 5. Output Head: 64 * 1000 = 64,000 (Wait, this is TIED to the embedding layer!)
    
    # Total distinct parameters = 64,000 + 99,456 + 128 = 163,584
    expected_params = 163584
    actual_params = model.get_num_params()
    
    assert actual_params == expected_params, f"Expected {expected_params} params, got {actual_params}"

def test_weight_tying(model):
    # The pointers to the underlying memory should be exactly the same
    emb_ptr = model.token_emb.embedding.weight.data_ptr()
    head_ptr = model.head.weight.data_ptr()
    
    assert emb_ptr == head_ptr, "Embedding and output head weights are not tied!"