# tinystories-gpt

Training a small language model from scratch on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), one step at a time: data pipeline → bigram baseline → attention → a small GPT.

Part of my 30-day "model user → model trainer" sprint. Tokenizer from [bpe-from-scratch](https://github.com/nehaai/bpe-from-scratch).

## Scoreboard

Val loss = average cross-entropy on stories the model never trained on. Lower is better. Every new model has to beat the one above it.

| Model | Params | Val loss | Notes |
|---|---:|---:|---|
| Uniform guess | 0 | 8.318 | ln(4096), the loss before any training |
| Bigram | 16.8M | **3.57** | one 4096×4096 lookup table, sees only the previous token |

## Data

- My own byte-level BPE, trained to 4,095 tokens on 20 MB of TinyStories. Id 4095 is reserved for end-of-story.
- **3.95 bytes/token** on TinyStories, against 2.77 when the same tokenizer was trained on *Pride and Prejudice*. Children's stories reuse a small vocabulary, so nearly every word becomes a single token: `Once| upon| a| time|,| there| was| a| little| girl| named| Lily|.`
- Train: first 200 MB of the train split → 49.7M tokens. Val: the full valid split → 5.6M tokens. Both saved as flat `uint16` arrays.

## Bigram baseline

The simplest model that can learn anything: look at the current token and predict the next one. It's one embedding table, where row *i* holds the scores for "what comes after token *i*".

```
step     0 | train 8.3178 | val 8.3178
step  1000 | train 3.7635 | val 3.7749
step  3000 | train 3.5693 | val 3.5908
step  5000 | train 3.5337 | val 3.5713
```

- **Step 0 = ln(4096).** The table is zero-initialised, so the first guess is uniform. Default `N(0,1)` init would start around 8.8.
- **Train ≈ val.** A lookup table is too simple to memorise 50M tokens.
- **Plateau at ~3.57.** e^3.57 ≈ 35, so at every token it's as unsure as picking from about 35 options. That's the ceiling for a model that sees one token of context.

What it writes:

> Once upon a little girl named Lily. She says. She water. Emma Do you had to disturled and he was looking for a time, "Yes, Lily watched it and played.

Every adjacent pair is plausible, but the whole is nonsense. After "a" it picks "little" (the most common follow-up), because it doesn't remember "Once upon". It invents "disturled" by chaining token pieces, and it swaps between Lily, Emma, Ben and Tom because it can't remember who the story is about. Fixing that needs context, which is what attention adds next.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data
curl -L -o data/valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
curl -L -o data/train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
python prepare.py   # trains tokenizer (~35s), writes data/train.bin + data/val.bin (~15s)
python bigram.py    # ~5 min on an M-series Mac
```

| File | What it does |
|---|---|
| `bpe.py` | Byte-level BPE tokenizer (from bpe-from-scratch) |
| `prepare.py` | Trains/loads the tokenizer, streams TinyStories, writes `uint16` token files |
| `bigram.py` | `get_batch`, bigram model, train/val loss logging, sampling |