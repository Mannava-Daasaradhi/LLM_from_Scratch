import json
from collections import defaultdict
from typing import Optional

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
        self._trained = False

    def train(self, text: str, vocab_size: int = 10000) -> None:
        """
        Train BPE on text. After this call:
          - self.vocab has vocab_size entries
          - self.merges has (vocab_size - initial_vocab_size) entries in order

        ALGORITHM:
        1. Initialize character vocabulary:
           - Add 4 special tokens (IDs 0-3)
           - Split text into words (split on whitespace, keep punctuation)
           - Prepend 'Ġ' (U+0120) to each word to mark word boundaries
             (this is the GPT-2 convention — 'hello' becomes 'Ġhello')
           - Build initial vocab: every unique character in the corpus + special tokens
           - Represent each word as a tuple of characters: ('Ġ','h','e','l','l','o')

        2. Count word frequencies:
           word_freqs: dict mapping word_tuple → int count

        3. Repeat until vocab_size reached:
           a. Count all adjacent pair frequencies across all words
              (weighted by word frequency)
           b. Find the most frequent pair (break ties alphabetically)
           c. Merge that pair into a new token everywhere in word_freqs
           d. Add new token to vocab, record merge rule

        4. Store final vocab and merges.

        COMPLEXITY NOTE:
        Naive implementation is O(vocab_size * corpus_size). This is fine for
        1MB of text and 10K vocab. Do not optimize prematurely.
        """
        # 1. Initialize character vocabulary and word frequencies
        words = text.split() # Splits on whitespace
        word_freqs = defaultdict(int)
        
        for word in words:
            word = 'Ġ' + word
            chars = tuple(word)
            word_freqs[chars] += 1
            
            # Add unique characters to vocab
            for char in chars:
                if char not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[char] = idx
                    self.id_to_token[idx] = char

        # 3. Repeat until vocab_size reached
        while len(self.vocab) < vocab_size:
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

        self._trained = True

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode text to token IDs using trained merge rules.

        ALGORITHM:
        1. Split text into words (same whitespace split as training)
        2. For each word:
           a. Start with character-level representation (with Ġ prefix)
           b. Apply merges in order (same order as training):
              - Scan word for the highest-priority merge
              - Apply it, repeat until no merges apply
        3. Convert tokens to IDs (use UNK_TOKEN ID for unknown tokens)
        4. If add_special_tokens: prepend BOS_TOKEN ID, append EOS_TOKEN ID
        5. Return list of int IDs

        IMPORTANT: Merges must be applied in training order, not greedily by
        frequency of the current text. This is what makes BPE deterministic.
        """
        ids = []
        if add_special_tokens:
            ids.append(self.vocab[self.BOS_TOKEN])
            
        words = text.split()
        for word in words:
            word = 'Ġ' + word
            word_tuple = tuple(word)
            
            # Apply merges in the exact order they were learned
            for merge_pair in self.merges:
                new_word_tuple = []
                i = 0
                while i < len(word_tuple):
                    if i < len(word_tuple) - 1 and word_tuple[i] == merge_pair[0] and word_tuple[i+1] == merge_pair[1]:
                        new_word_tuple.append(merge_pair[0] + merge_pair[1])
                        i += 2
                    else:
                        new_word_tuple.append(word_tuple[i])
                        i += 1
                word_tuple = tuple(new_word_tuple)
            
            # Convert to IDs
            for token in word_tuple:
                ids.append(self.vocab.get(token, self.vocab[self.UNK_TOKEN]))
                
        if add_special_tokens:
            ids.append(self.vocab[self.EOS_TOKEN])
            
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        1. Convert IDs to token strings
        2. Skip special tokens if skip_special_tokens=True
        3. Join tokens, replacing 'Ġ' with ' '
        4. Strip leading/trailing whitespace
        """
        tokens = []
        special_ids = {0, 1, 2, 3}
        
        for idx in ids:
            if skip_special_tokens and idx in special_ids:
                continue
            # Fallback to <unk> if ID doesn't exist
            tokens.append(self.id_to_token.get(idx, self.UNK_TOKEN))
            
        text = "".join(tokens)
        text = text.replace('Ġ', ' ')
        return text.strip()

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
        tokenizer._trained = True
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)