import json
import re
from collections import defaultdict
from functools import lru_cache
from typing import Optional


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    """
    GPT-2's reversible byte → unicode-char mapping.

    Byte-level BPE operates on bytes (0-255) so it can represent *any* text with no
    unknown tokens. But raw control bytes (newlines, tabs, …) are awkward to handle as
    vocabulary strings, so each byte is mapped to a single printable unicode char.
    Printable ASCII maps to itself; the rest map to code points starting at 256.
    Notably byte 0x20 (space) → 'Ġ' and 0x0A (newline) → 'Ċ' — which is why whitespace
    shows up as those glyphs in the vocab.
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {c: b for b, c in BYTE_ENCODER.items()}

# GPT-2-style pre-tokenization regex (stdlib `re`, Unicode-aware). Splits text into
# pieces that KEEP their leading whitespace, so word boundaries and newlines survive:
# contractions, " word", " 123", " punct", and runs of whitespace are each their own
# piece. `[^\W\d_]` approximates \p{L} (letters) under re.UNICODE.
PRETOKEN_RE = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _byte_encode(piece: str) -> str:
    """Map a raw text piece to its byte-level representation (one char per UTF-8 byte)."""
    return "".join(BYTE_ENCODER[b] for b in piece.encode("utf-8"))


def pretokenize(text: str) -> list[str]:
    """Split text into byte-encoded pre-token pieces (whitespace preserved)."""
    return [_byte_encode(m.group()) for m in PRETOKEN_RE.finditer(text)]


class BPETokenizer:
    # Special tokens — always in vocabulary at fixed IDs
    PAD_TOKEN = "<pad>"   # ID 0
    UNK_TOKEN = "<unk>"   # ID 1
    BOS_TOKEN = "<bos>"   # ID 2
    EOS_TOKEN = "<eos>"   # ID 3

    def __init__(self):
        # Initialize with special tokens
        self.vocab: dict[str, int] = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3
        }
        self.id_to_token: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merges: list[tuple[str, str]] = [] # ordered merge rules
        # merge_ranks: pair -> its priority (index in self.merges). Lower = applied first.
        # Lets encode() pick the next merge in O(word_len) instead of scanning all merges.
        self.merge_ranks: dict[tuple[str, str], int] = {}
        # Per-word BPE cache. The corpus has heavy word repetition, so caching the
        # tokenisation of each unique word is the single biggest encode speedup.
        self._cache: dict[str, tuple[str, ...]] = {}
        self._trained = False

    def train(self, text: str, vocab_size: int = 10000) -> None:
        """
        Train byte-level BPE on text. After this call:
          - self.vocab has vocab_size entries
          - self.merges has (vocab_size - initial_vocab_size) entries in order

        ALGORITHM:
        1. Pre-tokenise with a GPT-2-style regex that KEEPS whitespace (so newlines and
           word boundaries survive), then byte-encode each piece (UTF-8 byte → unicode
           char). Word boundaries appear as 'Ġ' (space) / 'Ċ' (newline) prefixes.
        2. Count piece frequencies: word_freqs maps a char-tuple → count.
        3. Initialise vocab with the 4 special tokens + every byte-char observed.
        4. Repeat until vocab_size:
           a. Count adjacent pair frequencies (weighted by piece frequency)
           b. Pick the most frequent pair (ties broken alphabetically)
           c. Merge it everywhere; record the merge rule and new token.

        COMPLEXITY: O(vocab_size * corpus_size). Fine for a few MB at 10K vocab; for
        large corpora, learn merges from a representative sample (see train_tokenizer.py).
        """
        # 1 & 2. Pre-tokenise (whitespace-preserving) and count piece frequencies
        word_freqs = defaultdict(int)
        for piece in pretokenize(text):
            chars = tuple(piece)
            word_freqs[chars] += 1

            # 3. Add every observed byte-char to the vocab
            for char in chars:
                if char not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[char] = idx
                    self.id_to_token[idx] = char

        # 3. Repeat until vocab_size reached
        while len(self.vocab) < vocab_size:
            if len(self.vocab) % 500 == 0:
                print(f"Training BPE... Vocab size: {len(self.vocab)} / {vocab_size}")
            # a. Count all adjacent pair frequencies
            pairs = defaultdict(int)
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pairs[(word_tuple[i], word_tuple[i+1])] += freq
            
            if not pairs:
                break # No more pairs to merge
            
            # b. Find the most frequent pair (break ties alphabetically by negative freq, then char ascending)
            best_pair = sorted(pairs.keys(), key=lambda p: (-pairs[p], p[0], p[1]))[0]
            
            # d. Add new token to vocab, record merge rule
            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.id_to_token[new_id] = new_token
            self.merges.append(best_pair)
            
            # c. Merge that pair into a new token everywhere in word_freqs
            new_word_freqs = defaultdict(int)
            for word_tuple, freq in word_freqs.items():
                new_word_tuple = []
                i = 0
                while i < len(word_tuple):
                    if i < len(word_tuple) - 1 and word_tuple[i] == best_pair[0] and word_tuple[i+1] == best_pair[1]:
                        new_word_tuple.append(new_token)
                        i += 2
                    else:
                        new_word_tuple.append(word_tuple[i])
                        i += 1
                new_word_freqs[tuple(new_word_tuple)] = freq
            
            word_freqs = new_word_freqs

        # Build the rank lookup used by the fast encoder.
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._cache = {}
        self._trained = True

    def _bpe(self, word: str) -> tuple[str, ...]:
        """
        Tokenise a single byte-encoded pre-token piece into BPE sub-tokens.

        Repeatedly merges the adjacent pair with the lowest merge rank until no
        adjacent pair is mergeable. This is mathematically equivalent to applying
        the learned merges in training order — a later merge can never recreate an
        earlier-ranked adjacency — but it runs in O(word_len^2) instead of
        O(num_merges) per word. Results are cached per unique word.
        """
        cached = self._cache.get(word)
        if cached is not None:
            return cached

        symbols = list(word)
        while len(symbols) >= 2:
            # Find the adjacent pair with the lowest (best) merge rank.
            best_rank = None
            best_i = -1
            for i in range(len(symbols) - 1):
                rank = self.merge_ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_rank is None:
                break  # nothing left to merge

            # Merge every non-overlapping occurrence of the chosen pair, L→R.
            a, b = symbols[best_i], symbols[best_i + 1]
            merged = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged

        result = tuple(symbols)
        self._cache[word] = result
        return result

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode text to token IDs using trained merge rules.

        1. Pre-tokenise (whitespace-preserving regex) and byte-encode each piece
        2. BPE-tokenise each piece (see _bpe), with per-piece caching
        3. Convert tokens to IDs (UNK only for byte-chars never seen in training)
        4. If add_special_tokens: prepend BOS, append EOS
        """
        # Self-heal merge_ranks if merges were set directly (e.g. in tests) rather
        # than via train()/load(), so the fast encoder always matches self.merges.
        if len(self.merge_ranks) != len(self.merges):
            self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
            self._cache = {}

        ids = []
        unk = self.vocab[self.UNK_TOKEN]
        if add_special_tokens:
            ids.append(self.vocab[self.BOS_TOKEN])

        for piece in pretokenize(text):
            for token in self._bpe(piece):
                ids.append(self.vocab.get(token, unk))

        if add_special_tokens:
            ids.append(self.vocab[self.EOS_TOKEN])

        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text — an exact inverse of encode for any text the
        tokenizer was trained on (whitespace and newlines included).
        1. Convert IDs to token strings (each is a run of byte-chars)
        2. Skip special tokens if requested
        3. Concatenate, map each byte-char back to its byte, decode UTF-8
        """
        special_ids = {0, 1, 2, 3}
        tokens = []
        for idx in ids:
            if skip_special_tokens and idx in special_ids:
                continue
            tokens.append(self.id_to_token.get(idx, self.UNK_TOKEN))

        byte_string = "".join(tokens)
        # Map each byte-char back to its byte; ignore any char that isn't a byte-char
        # (e.g. a leftover special-token glyph when skip_special_tokens=False).
        byte_values = bytes(BYTE_DECODER[c] for c in byte_string if c in BYTE_DECODER)
        return byte_values.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """
        Save tokenizer to JSON file:
        {
          "vocab": {"token": id, ...},
          "merges": [["tok_a", "tok_b"], ...],
          "version": "1.0"
        }
        """
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "version": "1.0"
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from JSON file saved by save()."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        tokenizer = cls()
        tokenizer.vocab = data["vocab"]
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.vocab.items()} # Ensure keys are ints, values are strings
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.merge_ranks = {pair: i for i, pair in enumerate(tokenizer.merges)}
        tokenizer._cache = {}
        tokenizer._trained = True
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)