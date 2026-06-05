# 🤖 Build a Chatbot From Scratch — BUILD PLAN

You are going to rebuild the entire LLM by hand and wrap it in something you can **talk to**.
By the end, `python chat.py` opens a prompt and a model *you* wrote — letter by letter —
talks back.

## The rules (you asked me to force you)

1. **No copy-paste.** Not from `../model`, not from the book, not from me. You rebuild each
   component from your `study/` sheets and your memory. This is the blank-file test at project
   scale.
2. **Gated stages.** Before you write a component, you must pass my **out-loud quiz** on it
   (closed book). Before you advance, your code must pass the stage's **success test**.
3. **Type every line yourself.** I explain, I quiz, I point at bugs. I never write your code.
4. When a stage passes, tick its box below and tell me — I'll quiz-gate the next one.

## The build order (bottom-up — each stage needs the one before it)

| Stage | You build | Success test | Study sheet |
|------|-----------|--------------|-------------|
| 0 | **Setup** — make the folders + `config.py` | folders exist; `config.py` loads | — |
| 1 | **Tokenizer** `tokenizer/bpe.py` (train, encode, decode) | `decode(encode(s)) == s` round-trips | `study/tokenization.md`* |
| 2 | **Embeddings** `model/embedding.py` (token emb + RoPE) | `(B,T)`→`(B,T,d_model)` | `study/embeddings.md`* |
| 3 | **Attention** `model/attention.py` | causal mask kills future; rows sum to 1 | `study/attention.md` ✅ |
| 4 | **Block** `model/block.py` + `feedforward.py` (RMSNorm + SwiGLU + residual) | shape in == shape out | `study/block.md`* |
| 5 | **Full model + loss** `model/transformer.py` | forward gives `(B,T,vocab)`; loss is scalar | `study/model_loss.md`* |
| 6 | **Training** `train.py` | loss drops on one batch over 50 steps | `study/training.md`* |
| 7 | **Generation** `generate.py` | produces text, not gibberish | `study/generation.md`* |
| 8 | **Chat loop** `chat.py` (multi-turn context + REPL) | holds a 3-turn conversation | `study/chat.md`* |
| 9 | *(stretch)* **Chat formatting / instruction tuning** | follows a simple instruction | book Ch 7 |

\* sheets marked `*` don't exist yet — I write each one as you reach its stage.

## Stage 0 — DO THIS NOW (your first task)

Create this structure inside `chatbot/` yourself (you make the folder = you own it):

```
chatbot/
├── config.py            # hyperparameters (you write it in Stage 0)
├── tokenizer/
│   └── bpe.py           # Stage 1
├── model/
│   ├── embedding.py     # Stage 2
│   ├── attention.py     # Stage 3
│   ├── feedforward.py   # Stage 4
│   ├── block.py         # Stage 4
│   └── transformer.py   # Stage 5
├── data/                # your training corpus goes here
├── train.py             # Stage 6
├── generate.py          # Stage 7
└── chat.py              # Stage 8  ← the part that makes it a chatbot
```

In PowerShell, from the repo root, you can scaffold the empty dirs/files yourself, e.g.:
`! mkdir chatbot\tokenizer, chatbot\model, chatbot\data`
(the `!` prefix runs it in this session so I see the result.)

## Progress

- [ ] Stage 0 — setup
- [ ] Stage 1 — tokenizer
- [ ] Stage 2 — embeddings
- [ ] Stage 3 — attention
- [ ] Stage 4 — block
- [ ] Stage 5 — model + loss
- [ ] Stage 6 — training
- [ ] Stage 7 — generation
- [ ] Stage 8 — chat loop
- [ ] Stage 9 — chat formatting (stretch)
```
```
