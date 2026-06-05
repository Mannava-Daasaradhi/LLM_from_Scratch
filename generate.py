"""
python generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "To be or not to be" \
  --max_tokens 200 \
  --temperature 0.8 \
  --top_p 0.9
"""
import argparse
from types import SimpleNamespace

import torch

from tokenizer.bpe import BPETokenizer
from model.transformer import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint (config is stored as a plain dict)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]

    # Build the model config from the stored dict. SimpleNamespace gives attribute
    # access (config.d_model, config.rope_theta, ...) without caring which optional
    # fields are present, so new keys like rope_theta just work.
    model_cfg = SimpleNamespace(**config["model"])
    tokenizer = BPETokenizer.load(config["data"]["tokenizer_path"])

    model = GPT(model_cfg)
    # Strip any torch.compile '_orig_mod.' prefix that may remain.
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state)
    model.to(device).eval()

    meta = []
    if "step" in ckpt:
        meta.append(f"step {ckpt['step']}")
    if "val_loss" in ckpt:
        meta.append(f"val_loss {ckpt['val_loss']:.4f}")
    if "val_accuracy" in ckpt:
        meta.append(f"acc {ckpt['val_accuracy'] * 100:.2f}%")
    if meta:
        print(f"[checkpoint: {', '.join(meta)}]")

    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)

    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    new_tokens = output_ids[0, input_ids.shape[1]:].tolist()
    generated_text = tokenizer.decode(new_tokens)
    print(f"\n=== PROMPT ===\n{args.prompt}")
    print(f"\n=== GENERATED ===\n{generated_text}")


if __name__ == "__main__":
    main()
