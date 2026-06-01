"""
Download and assemble a training corpus, then split into train/val.

The original project trained on ~1 MB of "tiny Shakespeare" — far too little to
justify a 10 000-token BPE vocabulary (most tokens were seen only a handful of
times). This script can instead assemble a much larger public-domain corpus from
Project Gutenberg so the vocabulary and the ~40 M-parameter model actually have
enough data to learn from.

Usage:
    python data/download.py                       # default: 'classics' (~25-35 MB)
    python data/download.py --dataset shakespeare # complete works only (~5 MB)
    python data/download.py --dataset tinyshakespeare
    python data/download.py --val-frac 0.1
"""
import argparse
import os
import re
import urllib.request

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# Project Gutenberg ebook IDs. Plain-text UTF-8 is served from /cache/epub/<id>/pg<id>.txt
SHAKESPEARE_IDS = [100]  # The Complete Works of William Shakespeare (~5.3 MB)

CLASSICS_IDS = [
    100,    # Shakespeare — Complete Works
    2600,   # Tolstoy — War and Peace
    2701,   # Melville — Moby Dick
    1342,   # Austen — Pride and Prejudice
    1661,   # Doyle — Adventures of Sherlock Holmes
    98,     # Dickens — A Tale of Two Cities
    1400,   # Dickens — Great Expectations
    84,     # Shelley — Frankenstein
    345,    # Stoker — Dracula
    1260,   # Brontë — Jane Eyre
    768,    # Brontë — Wuthering Heights
    174,    # Wilde — The Picture of Dorian Gray
    158,    # Austen — Emma
    2591,   # Grimm — Fairy Tales
    6130,   # Homer — The Iliad
    1727,   # Homer — The Odyssey
    1497,   # Plato — The Republic
    1232,   # Machiavelli — The Prince
    120,    # Stevenson — Treasure Island
    219,    # Conrad — Heart of Darkness
    36,     # Wells — The War of the Worlds
    35,     # Wells — The Time Machine
    1080,   # Swift — A Modest Proposal
    205,    # Thoreau — Walden
    2554,   # Dostoevsky — Crime and Punishment
]

_HEADERS = {"User-Agent": "Mozilla/5.0 (LLM_from_Scratch corpus builder)"}

# Project Gutenberg wraps each book in a license header/footer we want to strip.
_START_RE = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                       re.IGNORECASE | re.DOTALL)
_END_RE = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK",
                     re.IGNORECASE)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    text = raw.decode("utf-8", errors="ignore")
    return text.lstrip("﻿")  # strip BOM if present


def _strip_gutenberg_boilerplate(text: str) -> str:
    m = _START_RE.search(text)
    if m:
        text = text[m.end():]
    m = _END_RE.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def _gutenberg_url(book_id: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


def build_corpus(dataset: str) -> str:
    if dataset == "tinyshakespeare":
        print("Downloading tiny Shakespeare...")
        return _fetch(TINY_SHAKESPEARE_URL)

    ids = SHAKESPEARE_IDS if dataset == "shakespeare" else CLASSICS_IDS
    parts = []
    for book_id in ids:
        try:
            print(f"  fetching Gutenberg #{book_id} ...", end=" ", flush=True)
            text = _strip_gutenberg_boilerplate(_fetch(_gutenberg_url(book_id)))
            parts.append(text)
            print(f"{len(text):,} chars")
        except Exception as e:  # skip individual failures, keep going
            print(f"SKIPPED ({type(e).__name__}: {e})")
    if not parts:
        raise RuntimeError("No books downloaded — check your network connection.")
    # Two blank lines between books as a soft document separator.
    return "\n\n\n".join(parts)


def prepare_data(dataset: str = "classics", val_frac: float = 0.1, data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    input_path = os.path.join(data_dir, "input.txt")
    train_path = os.path.join(data_dir, "train.txt")
    val_path = os.path.join(data_dir, "val.txt")

    text = build_corpus(dataset)
    with open(input_path, "w", encoding="utf-8") as f:
        f.write(text)

    split_index = int(len(text) * (1.0 - val_frac))
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(text[:split_index])
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(text[split_index:])

    print(f"\nCorpus '{dataset}': {len(text):,} chars total")
    print(f"  Train: {split_index:,} chars | Val: {len(text) - split_index:,} chars")
    print(f"  Wrote {input_path}, {train_path}, {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the training corpus.")
    parser.add_argument("--dataset", choices=["tinyshakespeare", "shakespeare", "classics"],
                        default="classics", help="Which corpus to assemble (default: classics)")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction held out for validation (default: 0.1)")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    prepare_data(args.dataset, args.val_frac, args.data_dir)
