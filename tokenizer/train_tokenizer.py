"""
Train a BPE tokenizer from a text corpus.

    python tokenizer/train_tokenizer.py \
        --input data/train.txt \
        --vocab-size 10000 \
        --output tokenizer/shakespeare_bpe.json
"""
import argparse
import time

from tokenizer.bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer.")
    parser.add_argument("--input", default="data/train.txt",
                        help="Path to the training corpus (default: data/train.txt)")
    parser.add_argument("--vocab-size", type=int, default=10000,
                        help="Target vocabulary size (default: 10000)")
    parser.add_argument("--output", default="tokenizer/shakespeare_bpe.json",
                        help="Where to save the tokenizer JSON")
    parser.add_argument("--max-chars", type=int, default=0,
                        help="Learn merges from only the first N chars (0 = whole file). "
                             "BPE merges learned on a representative sample generalise to "
                             "the full corpus, which keeps training tractable on large files.")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        text = f.read()
    if args.max_chars and len(text) > args.max_chars:
        print(f"Loaded {len(text):,} chars from {args.input}; "
              f"training on first {args.max_chars:,}")
        text = text[:args.max_chars]
    else:
        print(f"Loaded {len(text):,} chars from {args.input}")

    tok = BPETokenizer()
    t0 = time.time()
    tok.train(text, vocab_size=args.vocab_size)
    print(f"Trained {tok.vocab_size}-token vocab in {time.time() - t0:.1f}s")
    tok.save(args.output)
    print(f"Saved tokenizer to {args.output}")

    # Round-trip sanity check
    test = "To be or not to be, that is the question."
    encoded = tok.encode(test)
    decoded = tok.decode(encoded)
    assert decoded.strip() == test.strip(), (
        f"Round-trip failed:\n  original: {test}\n  decoded:  {decoded}"
    )
    print(f"Round-trip OK | sample encode: {encoded[:10]}...")


if __name__ == "__main__":
    main()
