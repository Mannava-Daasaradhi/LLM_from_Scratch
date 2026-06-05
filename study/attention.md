# ✍️ Pen-&-Paper Notes — Self-Attention

> Handwrite this whole sheet. Where you see **✍️ DRAW**, draw the picture by hand — the
> picture is half the understanding. Where you see **🧠 WHY**, you must be able to say it
> out loud with no notes before you move on. Code refs point at *your* repo so you can see
> each equation living in `model/attention.py`.

---

## 0. Where attention sits in the whole flow

```
token IDs (B,T)
   │  TokenEmbedding            model/embedding.py
   ▼
x  (B, T, d_model)              ← the "residual stream"
   │
   ▼   ┌─────────────────────────── one Transformer block ───────────────────────────┐
   │   │  x = x + Attention( RMSNorm(x) )      ← THIS SHEET                            │
   │   │  x = x + SwiGLU(   RMSNorm(x) )                                               │
   │   └──────────────────────────────────────────────────────────────────────────────┘
   │      ...repeated n_layers times...
   ▼
RMSNorm → head (Linear) → logits (B, T, vocab) → softmax → next-token probs
```

Attention is the **only** place where positions talk to each other. Everything else
(embeddings, RMSNorm, SwiGLU, the head) treats each position independently.

**One-breath definition:** *Attention lets each token look back over all earlier tokens and
pull in a weighted mixture of their information — where the weights are decided by how well
each token's "query" matches every other token's "key".*

---

## 1. The cast of characters (Q, K, V)

For every token position we build **three** vectors out of its embedding `x`:

| Name | Symbol | Intuition | "Asking what?" |
|------|--------|-----------|----------------|
| Query | **q** | what *this* token is looking for | "I'm a verb — where's my subject?" |
| Key   | **k** | what *this* token offers / advertises | "I'm a noun, I could be a subject" |
| Value | **v** | the information this token will hand over if attended to | the actual content |

They are made by **one** linear projection, then split:

```
qkv = x @ W_qkv          W_qkv : (d_model) → (3·d_model)   no bias
q, k, v = split qkv into three (d_model) chunks
```

🧠 WHY one combined matrix? One big matmul is faster on the GPU than three small ones, and
mathematically identical to three separate `W_q, W_k, W_v`.

> 📍 Code: `attention.py:90-94` — `qkv = self.qkv(x)`, `q,k,v = qkv.chunk(3, dim=-1)`.

---

## 2. Splitting into heads

`d_model` is sliced into `n_heads` independent **heads**, each of width `d_k = d_model / n_heads`.

```
d_model = 512,  n_heads = 8   →   d_k = 64
```

Reshape each of q,k,v:  `(B, T, d_model)  →  (B, H, T, d_k)`

```
q.view(B, T, H, d_k).transpose(1,2)   # (B, H, T, d_k)
```

✍️ DRAW the reshape: a row of 512 numbers chopped into 8 blocks of 64, then stood up as 8
separate "lanes".

🧠 WHY multiple heads? Each head can learn a *different relationship* (one tracks
subject→verb, another tracks quotes→speaker, etc.). They run in parallel and get concatenated
back at the end. More heads = more relationship types, same total compute.

> 📍 Code: `attention.py:92-94`.

---

## 3. RoPE — injecting position (YOUR repo's choice)

> ⚠️ **Book vs your repo.** Raschka *adds* a position vector to the embedding once, at the
> bottom (learned absolute PE). **Your repo uses RoPE**: it *rotates* q and k by an angle
> proportional to their position, **inside attention, every layer.** Learn the book's version
> as the simple baseline, then defend RoPE.

**Idea:** give each position `m` a rotation angle. Rotate that token's q and k vectors by
that angle. When you later take a dot product `q_m · k_n`, the result depends only on the
**difference** `m − n` (relative position), because rotating both by their absolute angle and
dotting cancels the absolute part. That's the magic: absolute rotation in → relative position
out.

Per pair of dimensions `(2i, 2i+1)`, with frequency `θ_i = 1 / base^(2i/d_k)`:

```
angle at position m  =  m · θ_i
rotate the 2-D sub-vector (x_2i, x_2i+1) by that angle:

  [ x'_2i   ]   [ cos(mθ_i)  −sin(mθ_i) ] [ x_2i   ]
  [ x'_2i+1 ] = [ sin(mθ_i)   cos(mθ_i) ] [ x_2i+1 ]
```

In the code this is done for the whole vector at once with the identity
`x_rot = x·cos + rotate_half(x)·sin`, where `rotate_half([a,b]) = [−b, a]`.

🧠 WHY RoPE beats learned PE (say all three):
1. **Relative-position aware** — attention score depends on `m−n`, so the model generalises to
   distances/lengths it didn't see much in training.
2. **Zero parameters** — it's pure trig, nothing to learn (learned PE is a `max_seq × d_model`
   table of weights).
3. **Nothing added to the residual stream** — it touches only q and k *inside* attention, so
   the residual stream stays "clean" (this is why `TokenEmbedding` no longer multiplies by
   √d_model — see `embedding.py:18-29`).

> 📍 Code: `embedding.py:117-170` (`rotate_half`, `apply_rotary`, `RotaryEmbedding`) and
> `attention.py:96-100`. Note it's applied to **q and k only — never v.** (Quiz gotcha.)

---

## 4. Scaled dot-product attention — the core equation

The whole thing is ONE formula. Memorise it:

```
                  ┌            Q Kᵀ          ┐
Attention(Q,K,V) = softmax │  ──────  + mask  │  V
                  └           √d_k           ┘
```

Step by step, with shapes (per head):

```
1.  scores   = Q @ Kᵀ            (B,H,T,d_k)·(B,H,d_k,T) → (B,H,T,T)
2.  scores   = scores / √d_k     scale
3.  scores   = scores + mask     causal: future = −∞   (see §5)
4.  weights  = softmax(scores)   row-wise → (B,H,T,T), each row sums to 1
5.  out      = weights @ V       (B,H,T,T)·(B,H,T,d_k) → (B,H,T,d_k)
```

`scores[i][j]` = "how much should query at position **i** pay attention to key at position
**j**?" The softmax turns each **row** into a probability distribution over the keys.

> 📍 Code: the explicit version is `scaled_dot_product_attention()` at `attention.py:9-30`.
> The production path calls PyTorch's fused **flash** kernel
> `F.scaled_dot_product_attention(q,k,v, is_causal=True)` at `attention.py:111` — same math,
> done in one fused GPU op with O(T) memory instead of materialising the (T×T) matrix.

### 4a. 🧠 WHY divide by √d_k — *derive it, don't memorise it*

Assume each entry of q and k is independent, mean 0, variance 1.

```
score = q · k = Σ (i=1..d_k) q_i · k_i
each term q_i·k_i :  mean 0,  variance 1
sum of d_k independent terms →  variance = d_k,  std = √d_k
```

So raw scores have spread `√d_k`. With `d_k=64` that's std ≈ 8 — large. Feeding large numbers
into softmax makes it **saturate**: one weight ≈ 1, the rest ≈ 0, so the gradient through
softmax ≈ 0 → **the model can't learn**. Dividing by √d_k pulls the variance back to ≈ 1, so
softmax stays in its responsive region.

**If you don't scale:** training stalls / is unstable; attention collapses to one-hot too early.

---

## 5. Causal masking — no peeking at the future

A language model predicts token `t+1` from tokens `≤ t`. So position `i` must **not** attend to
any position `j > i` (the future). We enforce it *before* softmax:

```
scores[i][j] = −∞   for all j > i
softmax(−∞) = e^(−∞) / Σ = 0     ← that future key gets exactly 0 weight
```

✍️ DRAW the (T×T) score grid as a square. Shade the **upper triangle** (j>i) black = masked.
The lower triangle + diagonal stay open. It's a staircase.

```
        keys j→   0    1    2    3
queries  0      [ ok   ✗    ✗    ✗ ]
   i↓    1      [ ok   ok   ✗    ✗ ]
         2      [ ok   ok   ok   ✗ ]
         3      [ ok   ok   ok   ok ]
```

🧠 WHY add −∞ instead of deleting entries? Adding −∞ then softmax is differentiable, vectorised,
and the same op for every row — far faster on GPU than slicing. `softmax(x + (−∞)) = 0` exactly.

> 📍 Code: `is_causal=True` does this implicitly (`attention.py:111`). The explicit additive
> `−inf` mask is built only for the KV-cache / padding case at `attention.py:121-128`.

---

## 6. Merge heads + output projection

```
out  : (B, H, T, d_k)
      → transpose → (B, T, H, d_k)
      → reshape   → (B, T, d_model)      # concatenate the heads back
out = out @ W_o                          # (d_model→d_model) mixes the heads, no bias
```

🧠 WHY the final `W_o`? The heads were computed independently; `W_o` lets the model **mix**
what the different heads found into one combined message for the residual stream.

> 📍 Code: `attention.py:131-132`.

---

## 7. ✍️ WORKED EXAMPLE — do this by hand (single head, d_k = 2, T = 3)

Pick tiny vectors so the arithmetic is doable on paper. Verify every number yourself.

```
       q (query)        k (key)         v (value)
pos0:  [1, 0]           [1, 0]          [1, 0]
pos1:  [0, 1]           [1, 1]          [0, 1]
pos2:  [1, 1]           [0, 1]          [1, 1]
```

**Step 1 — scores = Q·Kᵀ** (dot every query with every key):

```
        k0   k1   k2
q0 →  [  1    1    0 ]      (q0·k0=1, q0·k1=1, q0·k2=0)
q1 →  [  0    1    1 ]
q2 →  [  1    2    1 ]
```

**Step 2 — scale by √d_k = √2 ≈ 1.414:**

```
q0 → [0.71, 0.71, 0.00]
q1 → [0.00, 0.71, 0.71]
q2 → [0.71, 1.41, 0.71]
```

**Step 3 — causal mask (−∞ above diagonal):**

```
q0 → [0.71,  −∞,   −∞ ]
q1 → [0.00, 0.71,  −∞ ]
q2 → [0.71, 1.41, 0.71]
```

**Step 4 — softmax each row** (e^x / Σ e^x; masked → 0):

```
q0 → [1.000, 0.000, 0.000]            (only one open entry)
q1 → [0.330, 0.670, 0.000]            (e^0=1, e^.71=2.03 → /3.03)
q2 → [0.248, 0.503, 0.248]            (e^.71, e^1.41, e^.71 → /8.17)
```
Check: each row sums to 1. ✓  Upper triangle is 0. ✓

**Step 5 — out = weights · V:**

```
q0 = 1·[1,0]                                   = [1.000, 0.000]
q1 = .330·[1,0] + .670·[0,1]                   = [0.330, 0.670]
q2 = .248·[1,0] + .503·[0,1] + .248·[1,1]      = [0.496, 0.751]
```

Read it back: **position 2's output is a blend of all three values, weighted by how much its
query matched each key** — and position 0 could only ever see itself. That single table *is*
attention. If you can reproduce it from blank paper, you understand the chapter.

---

## 8. Shape ledger (keep this in the margin)

```
x            (B, T, d_model)
qkv          (B, T, 3·d_model)
q,k,v        (B, T, d_model)   → heads → (B, H, T, d_k)
scores       (B, H, T, T)
weights      (B, H, T, T)      rows sum to 1
out (heads)  (B, H, T, d_k)    → merge → (B, T, d_model)
out (final)  (B, T, d_model)   ← same shape as input x  (so it can be added to residual)
```

B=batch, T=sequence length, H=n_heads, d_k=d_model/H.

---

## 9. Feynman gates — say each OUT LOUD, no notes, before Day 3

- [ ] State the one attention equation and name every symbol + its shape.
- [ ] Explain Q, K, V with a sentence each ("query asks…, key advertises…, value carries…").
- [ ] **Derive** why we divide by √d_k (variance argument) and what breaks without it.
- [ ] Explain the causal mask and why `softmax(−∞)=0` is the trick.
- [ ] Say why we use multiple heads.
- [ ] Reproduce the §7 worked example on blank paper.
- [ ] Explain RoPE: what it rotates, *where* it's applied (q & k, inside attention — not v,
      not the residual), and 3 reasons it beats learned PE.
- [ ] Point to the line in `attention.py` for each step above.

**Confidence I *understand* (not recognise) attention: ___ / 5**
**The one step that's still fuzzy:**
-
