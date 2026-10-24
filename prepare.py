"""Tokenize TinyStories with your BPE and save train.bin / val.bin as uint16.

Data (put both files in ./data/):
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

Build in this order:
  1. add save()/load() to your Tokenizer (you need the merges to survive between runs)
  2. train BPE to 4096 on a ~20 MB slice of train, save it, sanity-check a round-trip
  3. encode val (~22 MB) end to end -> val.bin
  4. encode a slice of train (start with ~200 MB, ~50M tokens) -> train.bin
"""
import sys
from functools import lru_cache
from array import array

import numpy as np

sys.path.append(".")  # adjust to wherever your repo lives
from bpe import Tokenizer

VOCAB = 4096
EOT_ID = VOCAB - 1   # 4095 = "story ended"  # TinyStoriesV2 separates stories with this literal string
EOT = "<|endoftext|>"     # the text that separates stories in the file

def save(tok, path):
    # TODO: write merges in order, one "a b" pair per line.
    # Why is it enough to save only the merges and not the vocab?
    with open(path, "w") as f:
        for a, b in tok.merges:          # dict order = training order
            f.write(f"{a} {b}\n")



def load(path):
    # TODO: rebuild tok.merges and tok.vocab from the file. The order of lines is the priority.
    tok = Tokenizer()
    with open(path) as f:
        for i, line in enumerate(f):
            a, b = map(int, line.split())
            new_id = 256 + i             # line number = token id
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
    return tok

def encode_file(tok, path, max_bytes=None):
    """Stream the file one story at a time and return a flat uint16 array.

    Speed: your _encode_chunk is pure Python, but TinyStories repeats the same
    few thousand words endlessly. Wrap it in a cache so each distinct chunk is
    encoded once. That's the difference between hours and minutes.
    """

    # TODO:
    #   - read stories split on EOT (don't load 2 GB with one .read() + findall)
    #   - for each story: ids = [i for c in tok._chunks(story) for i in enc_chunk(c.encode())]
    #   - append an end-of-story id between stories. Which id? You have no special
    #     token. Options: reserve id 4095 for EOT (train to 4095 merges), or use "\n\n".
    #     Pick one and write down why.
    #   - stop after max_bytes of text
    #   - assert max(ids) < 2**16 before casting, then return np.array(ids, dtype=np.uint16)
    
    # remember each word's tokens, so a word is only encoded once
    
    enc_chunk = lru_cache(maxsize=None)(lambda b: tuple(tok._encode_chunk(b)))
    ids = array("H")          # compact list of 2-byte numbers (a normal list would eat GBs of RAM)
    n_read, buf = 0, ""

    with open(path, encoding="utf-8") as f:
        while True:
            block = f.read(1 << 20)          # read 1 MB at a time, never the whole 2 GB
            if not block:
                break
            n_read += len(block)
            buf += block
            *stories, buf = buf.split(EOT)   # last piece may be half a story: keep it for next round
            for story in stories:
                story = story.strip()
                if story:
                    for c in tok._chunks(story):
                        ids.extend(enc_chunk(c.encode("utf-8")))
                    ids.append(EOT_ID)       # "story ended"
            print(f"\r  {n_read / 1e6:.0f} MB, {len(ids) / 1e6:.1f}M tokens", end="", flush=True)
            if max_bytes and n_read >= max_bytes:
                break
    print()
    assert max(ids) < 2**16
    return np.frombuffer(ids, dtype=np.uint16)


if __name__ == "__main__":
    import os, time

    # read the first ~20 MB of train: plenty to learn the merges
    with open("data/train.txt", encoding="utf-8") as f:
        sample = f.read(20_000_000)

    # remove the story separator so BPE doesn't learn junk merges for it
    sample = sample.replace(EOT, "\n")

    if os.path.exists("data/tiny.merges"):
        tok = load("data/tiny.merges")
        print("loaded tokenizer")
    else:
        t0 = time.time()
        tok = Tokenizer()
        tok.train(sample, VOCAB - 1)          # 4095 ids; 4095 itself is EOT
        save(tok, "data/tiny.merges")
        print(f"trained in {time.time() - t0:.0f}s")

    for split, path, limit in [("val", "data/valid.txt", None),
                               ("train", "data/train.txt", 200_000_000)]:
        t0 = time.time()
        arr = encode_file(tok, path, limit)
        arr.tofile(f"data/{split}.bin")
        print(f"{split}: {len(arr):,} tokens in {time.time() - t0:.0f}s")

    # sanity checks
    test = "Once upon a time, there was a little girl named Lily."
    ids = tok.encode(test)
    assert tok.decode(ids) == test
    print(len(ids), "tokens:", [tok.decode([i]) for i in ids])

    probe = sample[:1_000_000]
    print(f"bytes/token: {len(probe.encode()) / len(tok.encode(probe)):.2f}  (Austen was 2.77)")