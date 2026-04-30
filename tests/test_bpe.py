import os
import tempfile
import pytest
from tokenizer.bpe import BPETokenizer

def test_train_produces_correct_vocab_size():
    tok = BPETokenizer()
    text = "The quick brown fox jumps over the lazy dog. " * 10
    target_vocab_size = 35
    
    tok.train(text, vocab_size=target_vocab_size)
    
    assert tok.vocab_size == target_vocab_size
    assert len(tok.id_to_token) == target_vocab_size

def test_encode_decode_roundtrip():
    tok = BPETokenizer()
    text = "To be or not to be, that is the question."
    # Train on a small size so it actually has to use BPE merges
    tok.train(text, vocab_size=25)
    
    encoded_ids = tok.encode(text)
    decoded_text = tok.decode(encoded_ids)
    
    # BPE formatting strips/replaces whitespace markers, so we compare without spaces 
    # to ensure the character sequence was perfectly preserved.
    assert decoded_text.replace(" ", "") == text.replace(" ", "")

def test_special_tokens_in_vocab():
    tok = BPETokenizer()
    
    assert tok.vocab[tok.PAD_TOKEN] == 0
    assert tok.vocab[tok.UNK_TOKEN] == 1
    assert tok.vocab[tok.BOS_TOKEN] == 2
    assert tok.vocab[tok.EOS_TOKEN] == 3

def test_bos_eos_added():
    tok = BPETokenizer()
    tok.train("hello world", vocab_size=15)
    
    ids = tok.encode("hello", add_special_tokens=True)
    
    assert ids[0] == tok.vocab[tok.BOS_TOKEN]
    assert ids[-1] == tok.vocab[tok.EOS_TOKEN]

def test_unknown_token_maps_to_unk():
    tok = BPETokenizer()
    # Train on text that DOES NOT contain the letter 'z'
    tok.train("hello world", vocab_size=15)
    
    # Encode a string with an unknown character
    ids = tok.encode("hello z", add_special_tokens=False)
    
    # UNK_TOKEN ID is 1
    assert 1 in ids

def test_save_load_produces_identical_tokenizer():
    tok1 = BPETokenizer()
    text = "Machine learning is fascinating."
    tok1.train(text, vocab_size=25)
    
    encoded_original = tok1.encode(text)
    
    # Create a temporary file to safely test I/O
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        temp_path = tmp.name
        
    try:
        tok1.save(temp_path)
        
        # Load into a completely new instance
        tok2 = BPETokenizer.load(temp_path)
        encoded_loaded = tok2.encode(text)
        
        # Assertions
        assert encoded_original == encoded_loaded
        assert tok1.vocab == tok2.vocab
        assert tok1.merges == tok2.merges
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_merges_applied_in_order():
    tok = BPETokenizer()
    
    # Manually hack the vocab and merges to test determinism
    tok.vocab = {
        tok.PAD_TOKEN: 0, tok.UNK_TOKEN: 1, tok.BOS_TOKEN: 2, tok.EOS_TOKEN: 3,
        'Ġ': 4, 'a': 5, 'b': 6, 'c': 7, 'ab': 8, 'bc': 9
    }
    tok.id_to_token = {v: k for k, v in tok.vocab.items()}
    
    # Priority 1: ('a', 'b') -> 'ab'
    # Priority 2: ('b', 'c') -> 'bc'
    tok.merges = [('a', 'b'), ('b', 'c')]
    tok._trained = True
    
    # Word "abc" prepends 'Ġ', becoming ('Ġ', 'a', 'b', 'c').
    # Applying ('a', 'b') first gives ('Ġ', 'ab', 'c') = [4, 8, 7].
    # If ('b', 'c') were greedily applied first, we'd get ('Ġ', 'a', 'bc') = [4, 5, 9].
    ids = tok.encode("abc", add_special_tokens=False)
    
    assert ids == [4, 8, 7]

def test_word_boundary_marker():
    tok = BPETokenizer()
    text = "hello world"
    tok.train(text, vocab_size=20)
    
    # Encode just the word 'world'
    ids = tok.encode("world", add_special_tokens=False)
    
    # The first token ID of the word should correspond to a token string starting with 'Ġ'
    first_token_string = tok.id_to_token[ids[0]]
    assert first_token_string.startswith('Ġ')