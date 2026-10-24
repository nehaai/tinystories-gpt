"""Bigram baseline: one embedding table, logits = table[current_token].

Done when: a val loss number is written in the README. Every later model must beat it.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 4096
EOT_ID = VOCAB - 1
B, T = 64, 128
STEPS, EVAL_EVERY, EVAL_ITERS = 5000, 500, 50
LR = 1e-2  # a lookup table tolerates a big LR; transformers won't
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(1337)

# memmap = the OS reads from disk on demand, so the .bin files never sit fully in RAM
data = {s: np.memmap(f"data/{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}


def get_batch(split):
    d = data[split]
    ix = torch.randint(len(d) - T - 1, (B,))                    # B random start positions
    x = torch.stack([torch.from_numpy(d[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + T + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)                           # y = x shifted one to the left


class Bigram(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.table = nn.Embedding(V, V)  # row i = scores for "what comes after token i"
        nn.init.zeros_(self.table.weight)  # all scores equal -> uniform guess -> loss = ln(V)

    def forward(self, x, y=None):
        logits = self.table(x)  # (B, T, V)
        if y is None:
            return logits, None
        Bb, Tt, V = logits.shape
        loss = F.cross_entropy(logits.view(Bb * Tt, V), y.view(Bb * Tt))
        return logits, loss


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = [model(*get_batch(split))[1].item() for _ in range(EVAL_ITERS)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def generate(model, n=80):
    idx = torch.tensor([[EOT_ID]], device=device)  # start as if a story just ended
    for _ in range(n):
        logits, _ = model(idx[:, -1:])                 # a bigram only needs the last token
        probs = F.softmax(logits[:, -1, :], dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx[0].tolist()


if __name__ == "__main__":
    model = Bigram(VOCAB).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    print(f"device {device} | expected step-0 loss {math.log(VOCAB):.3f}")

    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0:
            l = estimate_loss(model)
            print(f"step {step:5d} | train {l['train']:.4f} | val {l['val']:.4f}")
        if step == STEPS:
            break
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # see what it writes
    from prepare import load
    tok = load("data/tiny.merges")
    ids = generate(model)
    print("\n" + "".join("\n---\n" if i == EOT_ID else tok.decode([i]) for i in ids[1:]))