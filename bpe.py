"""Byte-level BPE tokenizer, from scratch.

Fill in the TODOs. Build in this order and test each step before moving on:
  1. get_stats + merge  (test on a list like [1, 2, 3, 1, 2])
  2. train with pattern=None on a small string; print every merge
  3. decode, then encode; check decode(encode(x)) == x
  4. switch on the GPT-2 pre-split and retrain on the novel
"""
from collections import Counter

import regex as re  # pip install regex  (stdlib `re` has no \p{L} / \p{N})

# GPT-2's pre-split pattern. Read it piece by piece:
#   contractions | optional space + letters | optional space + digits |
#   optional space + punctuation | trailing whitespace | other whitespace
GPT2_SPLIT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(ids, counts=None, weight=1):
    """Count adjacent pairs in `ids`.

    Adds into `counts` if given (so you can accumulate across chunks), and
    adds `weight` per occurrence (so a chunk that appears 500 times in the
    text only needs to be scanned once).
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = Counter() if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] += weight
    return counts


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`.

    Example: merge([1, 2, 3, 1, 2], (1, 2), 256) -> [256, 3, 256]
    Edge case to think about: what should merge([5, 5, 5], (5, 5), 256) return?
    """
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2          # consumed both halves of the pair
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, pattern=GPT2_SPLIT):
        # pattern=None means "no pre-split": BPE runs over the raw byte stream.
        self.pattern = re.compile(pattern) if pattern else None
        self.merges = {}  # (id, id) -> new_id ; dict order == merge order == priority
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> raw bytes

    def _chunks(self, text):
        return self.pattern.findall(text) if self.pattern else [text]

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Speed tip: a novel has ~100k chunks but only ~10k *distinct* ones.
        # Count them, train on the distinct ones, weight pair counts by frequency.
        chunk_counts = Counter(self._chunks(text))
        words = {chunk: list(chunk.encode("utf-8")) for chunk in chunk_counts}

        for i in range(num_merges):
            new_id = 256 + i
 
            # 1. count pairs across all words, weighted by chunk frequency
            counts = Counter()
            for chunk, ids in words.items():
                get_stats(ids, counts, chunk_counts[chunk])
 
            # 2. nothing left to merge (every chunk is a single token)
            if not counts:
                break
 
            # 3. most frequent pair
            best = max(counts, key=counts.get)
 
            # 4. apply the merge everywhere
            for chunk in words:
                words[chunk] = merge(words[chunk], best, new_id)
 
            # 5. record it
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
 
            # 6. log
            if verbose:
                print(f"merge {i + 1}/{num_merges}: {best} -> {new_id} "
                      f"{self.vocab[new_id]!r} (count {counts[best]})")

    def decode(self, ids):
        # TODO: look up each id's bytes, join, decode as UTF-8.
        # Think: why might errors="replace" be needed for an arbitrary id list,
        # even though it never matters for decode(encode(x))?
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def _encode_chunk(self, chunk_bytes):
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            # the pair learned earliest (lowest merge id) goes first --
            # merges must be replayed in training order
            pair = min(zip(ids, ids[1:]),
                       key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        ids = []
        for chunk in self._chunks(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids