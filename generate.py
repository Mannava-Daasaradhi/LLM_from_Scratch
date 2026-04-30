"""
python generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "To be or not to be" \
  --max_tokens 200 \
  --temperature 0.8 \
  --top_p 0.9
"""
import torch, argparse
from tokenizer.bpe import BPETokenizer
from model.transformer import GPT
from config import load_config

class ConfigNode:
    """Helper class to access dict keys via dot notation"""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, ConfigNode(v) if isinstance(v, dict) else v)
            
    # Add these two methods so it behaves like a dictionary too!
    def __getitem__(self, key):
        return getattr(self, key)
        
    def keys(self):
        return self.__dict__.keys()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config_dict = ckpt["config"]

    from dataclasses import dataclass
    @dataclass
    class ModelConfig:
        vocab_size: int; d_model: int; n_heads: int; n_layers: int
        d_ff: int; max_seq_len: int; dropout: float

    model_cfg = ModelConfig(**config_dict['model'])
    tokenizer = BPETokenizer.load(config_dict['data']['tokenizer_path'])
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(args.prompt)]).long()

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Decode and print (only the new tokens)
    new_tokens = output_ids[0, input_ids.shape[1]:].tolist()
    generated_text = tokenizer.decode(new_tokens)
    print(f"\n=== PROMPT ===\n{args.prompt}")
    print(f"\n=== GENERATED ===\n{generated_text}")

if __name__ == "__main__":
    main()