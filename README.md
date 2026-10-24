# bpe-from-scratch

A byte-level BPE tokenizer in plain Python, the same algorithm GPT-2/GPT-4 tokenizers use, trained on *Pride and Prejudice* and compared against OpenAI's `tiktoken`.

Part of my 30-day "model user → model trainer" sprint.

## What's here

| File | What it does |
|---|---|
| `bpe.py` | `Tokenizer` with `train(text, vocab_size)`, `encode(str)`, `decode(ids)` (~120 lines, no dependencies beyond `regex`) |
| `check.py` | Trains to vocab 1,000, asserts `decode(encode(x)) == x` on edge cases, compares splits with GPT-2 / cl100k / o200k |
| `explore.py` | Two probes that explain why `12345` tokenizes the way it does |

## How BPE works, in one paragraph

Models only read numbers, so text has to be chopped into pieces ("tokens") that each get an id. Start with the 256 possible bytes, so any text in any language is representable. Then repeatedly find the most frequent adjacent pair and glue it into a new token. After 744 merges you have a 1,000-token vocabulary where common chunks (` the`, ` Elizabeth`) are single tokens and rare words fall back to smaller pieces. Encoding replays the merges **in the order they were learned**.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
curl -o novel.txt https://www.gutenberg.org/cache/epub/1342/pg1342.txt
python check.py novel.txt          # GPT-2 regex pre-split (~7s)
python check.py novel.txt --raw    # no pre-split (~45s), see the ugly tokens
python explore.py
```

## Results

All round-trips pass. Token counts for the same strings:

| Input | Mine (Austen, 1k vocab) | GPT-2 | cl100k | o200k |
|---|---:|---:|---:|---:|
| `12345 + 678` | 11 | 5 | 5 | 5 |
| A 3-line Python function | 37 | 26 | 18 | 18 |
| `मुझे हिंदी में बात करना पसंद है।` | **84** | 50 | 32 | **9** |

### 1. A tokenizer only knows what it was trained on

- **Numbers:** my tokenizer learned exactly one digit token (` 1`) from the whole novel, so `12345` stays as `1|2|3|4|5`.
- **Code:** no `def`, no `"""`, and Austen wrote no Python.
- **Hindi:** zero merges for Devanagari, so it falls back to raw bytes (3 bytes per character). o200k does the same sentence in 9 tokens. That's a **~9× token tax**: more cost and less context for the same sentence on a tokenizer that didn't see the language.

### 2. Why `12345` splits the way it does

1. **The regex cuts first.** The GPT-2 pattern hands BPE `['12345', ' +', ' 678']`, and merges can never cross those boundaries.
2. **The training data decides the merges.** Mine has no digit merges, so single digits. GPT-2 learned `123` and `45` from web text, but because digit runs are unlimited and a leading space attaches, ` 678` becomes ` 6|78`, so the same digits split inconsistently.
3. **Newer tokenizers force structure.** cl100k/o200k use `\p{N}{1,3}` and detach the space, so numbers are always chunked into groups of three from the left: `123|45`, `678`.

### 3. Why pre-split at all

Without the regex (`--raw`), BPE learns tokens like `ing the `, `she was `, `, but `, `that\n`. Compression on the novel is actually *better* (2.93 vs 2.77 bytes/token), but the vocabulary gets spent on this one book's phrases, the same word gets different ids depending on its neighbours, and common words like `return` fall apart (`re|tur|n `). Small loss in compression, much cleaner and more reusable tokens.

## What I'd try next

- Train on mixed English + Hindi + code and watch the token tax shrink
- Use right-to-left 3-digit grouping for numbers (`12|345`) and test whether it matters
- Make training faster: incremental pair counts instead of recounting every merge


   | Model  | Params | Val loss |
   |--------|-------:|---------:|
   | Bigram | 16.8M  | 3.57     |