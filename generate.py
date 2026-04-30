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
    config = ckpt["config"]
    tokenizer = BPETokenizer.load(config.data.tokenizer_path)
    model = GPT(config.model)
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