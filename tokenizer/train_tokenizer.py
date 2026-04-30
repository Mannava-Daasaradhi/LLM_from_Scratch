from tokenizer.bpe import BPETokenizer

with open("data/train.txt") as f:
    text = f.read()

tok = BPETokenizer()
tok.train(text, vocab_size=10000)
tok.save("tokenizer/shakespeare_bpe.json")

# Sanity checks
test = "To be or not to be, that is the question."
encoded = tok.encode(test)
decoded = tok.decode(encoded)
assert decoded.strip() == test.strip(), f"Round-trip failed:\n  original: {test}\n  decoded:  {decoded}"
print(f"Vocab size: {tok.vocab_size}")
print(f"Encoded: {encoded[:10]}...")
print(f"Decoded: {decoded}")
print("Tokenizer training complete.")