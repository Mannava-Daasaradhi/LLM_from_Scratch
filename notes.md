# Feynman Log — LLM_from_Scratch · Week 1

> **The contract (read every morning):**
> - AI **autocomplete OFF** while studying. AI may *explain* a concept ("why scale by √d?"); it may **never write code for me**.
> - I **type every line by hand** from the book. No copy-paste. My fingers learn what my eyes skim.
> - Each concept passes only when I can explain it **out loud, no notes, to an imaginary beginner**. If I can't say it simply, I don't understand it yet — that's the page to reread.
> - End of each day: the **blank-file test** — rebuild the day's core idea from an *empty file*, AI off.
>
> **Week 1 goal:** understand *how an LLM works end-to-end* AND understand *why my own `LLM_from_Scratch` repo is built the way it is*.
> **Week 1 final deliverable (Day 6–7):** implement self-attention in ~20 lines from scratch; explain Q/K/V, the √d scaling, and causal masking; walk through my repo's architecture and justify every modern choice (RoPE, RMSNorm, SwiGLU, flash attention) vs the GPT-2 baseline Raschka builds.
>
> **Key framing:** Raschka builds a **GPT-2-style** model (learned absolute positions, LayerNorm, GELU, vanilla attention). My repo is the **LLaMA-style upgrade** (RoPE, RMSNorm, SwiGLU, flash SDPA). Raschka = the skeleton. My repo = the upgraded body. Learn the skeleton first, then I can defend every upgrade.

---

# DAY 1 — June 2 — Text → Tokens → Embeddings
### Reading: Raschka Ch 1 (the big picture) + Ch 2 (working with text data)

**Today's one-sentence mission:** understand the *full path* from a raw string to the tensor that actually enters the transformer — `text → tokens → token IDs → token embeddings → (+ positional info) → input tensor` — well enough to rebuild it from a blank file and to explain how my repo's version differs.

## ⏱️ Time plan (8-hour sprint day)

| Block | Time | Activity |
|------|------|----------|
| 1 | 0:00–1:00 | **Read Ch 1** actively (pen in hand). No code. Goal: the mental model of what an LLM is and the train→finetune lifecycle. Answer §A below as you go. |
| 2 | 1:00–3:00 | **Read Ch 2** (tokenization, BPE, input-target pairs, token embeddings, positional embeddings). Type the book's small snippets by hand as they appear. Answer §B. |
| 3 | 3:00–3:45 | **Break + first recall pass.** Close the book. Try to say the whole `text→input tensor` path out loud. Note where you stalled. |
| 4 | 3:45–5:15 | **Map to MY repo** (§C). Open `tokenizer/bpe.py` and `model/embedding.py`. Hunt for the specific things listed. This is where "I vibecoded it" turns into "I understand it." |
| 5 | 5:15–6:45 | **Hand-coding exercises** (§D). Type from scratch — a toy tokenizer + the input/target windowing + an embedding lookup. AI off. |
| 6 | 6:45–7:30 | **Feynman write-up** — fill in §A/§B/§C answer slots in full sentences, in your own words. |
| 7 | 7:30–8:00 | **Blank-file test** (§E) + fill the **end-of-day rubric** (§F) + write tomorrow's questions. |

---

## §A — Ch 1: The Big Picture (fill in tonight, your own words)

1. **What is an LLM literally predicting**, one token at a time? Write the next-token-prediction objective as a sentence, then as a probability: `P(______ | ______)`.
   -

2. **Why "large"?** What two things scaled up (vs older NLP models) and why did scale unlock emergent ability rather than just lower error?
   -

3. **The lifecycle.** Define each stage in one line and say what data each uses:
   - Pretraining (self-supervised) —
   - Supervised fine-tuning (SFT) —
   - Alignment (RLHF / preference tuning) —
   - Why is pretraining called "self-supervised" when there are no human labels?
   -

4. **Where does the training signal come from** in pretraining if nobody labeled the data? (Hint: the label is hiding inside the text itself.)
   -

5. **The transformer's role.** Raschka's Fig 1.x shows GPT is *decoder-only*. In one sentence: what does "decoder-only / causal" mean and why is it the right shape for next-token prediction?
   -

6. **Emergent abilities & zero/few-shot.** What is in-context learning, and why is it surprising that a model trained *only* to predict the next token can follow instructions it was never explicitly trained on?
   -

## §B — Ch 2: Text → Embeddings (the core of today)

**B1. Tokenization basics**
1. Why can't we just feed raw characters or whole words into the model? State the failure mode of *each*: char-level (too ___), word-level (vocab too ___, can't handle ___).
   -
2. Walk the pipeline in order and define each step: **normalize → pre-tokenize (split) → encode to IDs → (later) decode back to text.**
   -

**B2. Byte-Pair Encoding (BPE) — the piece most people can't explain**
1. Describe the BPE *training* algorithm as a loop: start from ____, repeatedly find the ____ ____ pair and ____ it, until ____. What is a "merge"?
   -
2. Describe BPE *encoding* (applying a trained vocab to new text): given the learned merges, how do you turn "lowest" into tokens?
   -
3. Why does BPE elegantly solve the **out-of-vocabulary** problem — i.e., why is there essentially no `<unk>` for a byte-level BPE?
   -
4. Trade-off question: more merges (bigger vocab) → ____ tokens per sentence but ____ embedding table. Fewer merges → the opposite. Where's the tension?
   -

**B3. From token IDs to training examples**
1. **Input–target pairs.** For the token sequence `[A, B, C, D]` with context length 3, write out the (input window, target window) the way Raschka's sliding window produces them. Why is the target just the input **shifted by one**?
   - input:           target:
2. **Why a sliding window / stride?** What does `max_length` (context size) control, and what does `stride` change about how much the windows overlap?
   -
3. **Batching.** What are the two tensor dimensions of a batch that goes into the model, and what does each index mean? `(____, ____)`.
   -

**B4. Token embeddings**
1. What *is* the embedding matrix — its shape in terms of `vocab_size` and `d_model`/`emb_dim`, and what does a single row represent?
   -
2. An embedding "lookup" is mathematically equivalent to what matrix operation on a one-hot vector? Why do we use a lookup (indexing) instead of that matmul in practice?
   -
3. Are embeddings fixed (like word2vec downloaded weights) or learned during training here? What learns them?
   -

**B5. Positional embeddings**
1. **Why are they needed at all?** State the property of self-attention that makes it "blind to order" (permutation-______). What breaks if you remove positions entirely?
   -
2. Raschka adds a **learned absolute** positional embedding. What shape is it (in terms of `context_length` and `emb_dim`), and how is it combined with the token embedding — added or concatenated?
   -
3. What is the input tensor that finally enters block 1? Write the equation: `X = ____ + ____`.
   -

---

## §C — Map to MY repo (vibecode → understanding)

> Open the files. For each, find the lines, read them slowly, and write what they do **without** running anything. The point is to recognize today's Raschka concepts living in your own code — and to spot every place your code went *beyond* the book.

**C1. `tokenizer/bpe.py` + `tokenizer/train_tokenizer.py`**
- Find where merges are *learned* (training) vs *applied* (encode). Which function is which?
  -
- My repo uses **byte-level BPE + a GPT-2-style regex pre-tokenizer**, where Raschka Ch 2 just calls `tiktoken`. Two questions:
  - What does "byte-level" mean — what is the base alphabet before any merges? (Hint: how many possible bytes?)
    -
  - Why does byte-level give an **exact encode→decode round-trip** for *any* string, including newlines and emoji, with no `<unk>`?
    -
- The README brags the encode is now "rank-based + per-word cache" instead of O(words × merges). In plain words: what was slow before, and what's the trick now? (You don't need to optimize it — just explain it.)
  -

**C2. `model/embedding.py`**
- Find the **token embedding** layer (an `nn.Embedding`). Confirm its two size arguments map to `(vocab_size, d_model)`. Write the actual numbers from `configs/shakespeare.yaml` (`vocab_size: 10000`, `d_model: 512`): the table is `____ × ____ = ____` parameters.
  -
- **The big divergence:** Raschka uses *learned absolute* positional embeddings. My repo uses **RoPE (rotary)**. Find the RoPE code. Then answer:
  - What does RoPE *rotate*, and *where* is it applied — to the embeddings before the block, or to **Q and K inside attention**? (This is a classic interview gotcha — verify by reading, don't guess.)
    -
  - Name two concrete advantages the README claims for RoPE over learned PE ("relative-position aware", "zero params", "nothing added to the residual stream") — and explain *each* in your own words.
    -
- **Weight tying:** the README says token embedding ↔ LM head are tied. Find it. Why does sharing that matrix save ~5M params and *why might it even help* generalization?
  -

**C3. One honest sentence:** which part of C1/C2 did I genuinely not understand before today?
  -

---

## §D — Hand-coding exercises (type by hand, AI OFF)

> Do NOT copy from your own repo or the book. Derive it. Small toy scale is the point — understanding, not performance. Put these in a scratch file `day1_scratch.py` (delete after; the *learning* is the artifact).

1. **Toy tokenizer (char-level is fine here):** build `stoi`/`itos` dicts from a small string, write `encode(s) -> list[int]` and `decode(ids) -> str`. Assert `decode(encode(s)) == s`.
2. **Input/target windowing:** given a token list and `context_length`, produce `(x, y)` pairs where `y` is `x` shifted by one. Print 3 pairs and eyeball that the target really is the next token.
3. **Embedding lookup by hand:** make a random `nn.Embedding(vocab_size, d_model)` (or a raw tensor), feed a batch of token IDs shaped `(B, T)`, and confirm the output is `(B, T, d_model)`. Explain in a comment why the trailing dim appeared.
4. **Add positions:** create a `(T, d_model)` position tensor, add it to the token embeddings via broadcasting, and confirm the shape is unchanged. Write a one-line comment on *why broadcasting works here*.

**Success = all four run, and I can say what every line does without looking.**

---

## §E — Blank-file test (end of day, AI off, ~15 min)

Open a brand-new empty file. From memory, write code that:
1. Turns a short string into `input_ids` and `target_ids` (target shifted by one), and
2. Looks up their token embeddings and adds a positional tensor, printing the final `(B, T, d_model)` shape.

Did it run without peeking at notes/repo/book? **(y / n):**
If no — which step blocked me? That exact step is tomorrow's warm-up.

---

## §F — End-of-Day Rubric (be honest — green only if true)

- [ ] I can state the next-token objective as a probability, out loud, no notes.
- [ ] I can explain BPE training (merge loop) AND encoding, and why byte-level has no `<unk>`.
- [ ] I can produce (input, target) pairs by hand and explain the shift-by-one.
- [ ] I can state the embedding matrix shape and what one row means.
- [ ] I can explain why positions are needed (attention is permutation-invariant).
- [ ] I found token-embedding + RoPE in my own repo and can explain how RoPE differs from learned PE.
- [ ] Blank-file test: **passed**.

**Today's honest confidence (1–5) that I *understand* (not just recognize) this material:** ___ / 5

**The one thing I struggled with most:**
-

**Questions to resolve tomorrow (carry forward):**
-

---
<!-- DAY 2 starts Raschka Ch 3 (attention). Copy the Day-1 structure: Time plan → §A concept prompts → §C repo map → §D hand-coding → §E blank-file → §F rubric. Day 2 repo targets: model/attention.py (Q/K/V projections, the √d scaling, causal masking, flash SDPA), model/block.py (pre-norm residual structure). -->
