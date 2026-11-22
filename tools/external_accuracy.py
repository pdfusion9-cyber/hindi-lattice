#!/usr/bin/env python3
"""External, no-answer-injection accuracy audit for one Loop10 checkpoint."""
import argparse
import hashlib
import json
import mmap
import os
import random
import re
import struct
from collections import Counter, defaultdict

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True, help="The exact training GPACK5 artifact")
ap.add_argument("--ckpt", required=True, help="The checkpoint produced from that pack")
ap.add_argument("--test-parquet", required=True, help="A different Hindi FineWeb-2 shard")
ap.add_argument("--nq", type=int, default=500)
ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--nbr", type=int, default=8)
ap.add_argument("--cand", type=int, default=50)
ap.add_argument("--pat-cand", type=int, default=20)
ap.add_argument("--minpat", type=int, default=5)
a = ap.parse_args()
torch.set_num_threads(a.threads)
random.seed(1729)


class Pack:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        if self.mm[:6] != b"GPACK5":
            raise SystemExit("Training pack is not GPACK5")
        (self.V, self.ns, self.nch, self.cbits, self.missed,
         self.nblk, self.block, self.nlen) = struct.unpack("<IIIIIIII", self.mm[6:38])
        off = 38
        self.part = {}
        for name in ("chars", "voc", "lat", "latidx", "starts", "body", "blk", "lens"):
            (n,) = struct.unpack("<I", self.mm[off:off+4])
            off += 4
            self.part[name] = (off, n)
            off += n
        co, cn = self.part["chars"]
        self.chars = self.mm[co:co+cn].decode("utf-8")
        self._words = None

    @staticmethod
    def _vint(data, i):
        n = 0
        shift = 0
        while True:
            b = data[i]
            i += 1
            n |= (b & 0x7F) << shift
            if not (b & 0x80):
                return n, i
            shift += 7

    def words(self):
        if self._words is not None:
            return self._words
        off, size = self.part["voc"]
        data = self.mm[off:off+size]
        out, cur, acc, bits = [], [], 0, 0
        stop = len(self.chars)
        for byte in data:
            for k in range(8):
                acc |= ((byte >> k) & 1) << bits
                bits += 1
                if bits == self.cbits:
                    if acc == stop:
                        out.append("".join(cur))
                        cur = []
                    else:
                        cur.append(self.chars[acc])
                    acc = 0
                    bits = 0
        if cur:
            out.append("".join(cur))
        self._words = out
        return out

    def links(self, w):
        lo, _ = self.part["lat"]
        io, _ = self.part["latidx"]
        (x,) = struct.unpack("<I", self.mm[io+w*4:io+w*4+4])
        (y,) = struct.unpack("<I", self.mm[io+(w+1)*4:io+(w+1)*4+4])
        data = self.mm[lo+x:lo+y]
        count, i = self._vint(data, 0)
        out = []
        for _ in range(count):
            value, i = self._vint(data, i)
            out.append(value)
        return out

    def _bits(self, bitpos, nbits):
        off, _ = self.part["body"]
        value = 0
        for i in range(nbits):
            pos = bitpos + i
            value |= ((self.mm[off+(pos >> 3)] >> (pos & 7)) & 1) << i
        return value

    def path(self, idx):
        if idx < 0 or idx >= self.ns:
            return None
        bo, _ = self.part["blk"]
        so, _ = self.part["starts"]
        lo, _ = self.part["lens"]
        (bp,) = struct.unpack("<I", self.mm[bo+idx*4:bo+idx*4+4])
        (wid,) = struct.unpack("<I", self.mm[so+idx*4:so+idx*4+4])
        length = self.mm[lo+idx]
        out = [wid]
        for _ in range(length):
            links = self.links(wid)
            width = max(1, (max(len(links), 2)-1).bit_length())
            choice = self._bits(bp, width)
            bp += width
            if choice >= len(links):
                break
            wid = links[choice]
            out.append(wid)
        return out


class Model(nn.Module):
    def __init__(self, vocab_size, pattern_count, dim, layers, heads, nbr, use):
        super().__init__()
        self.d, self.H, self.K, self.use = dim, heads, nbr, use
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(256, dim)
        self.pat = nn.Embedding(max(pattern_count, 1), dim)
        self.blk = nn.ModuleList()
        for _ in range(layers):
            block = nn.ModuleDict({
                "n1": nn.LayerNorm(dim),
                "qkv": nn.Linear(dim, 3*dim, bias=False),
                "out": nn.Linear(dim, dim, bias=False),
                "n2": nn.LayerNorm(dim),
                "ff1": nn.Linear(dim, 4*dim),
                "ff2": nn.Linear(4*dim, dim),
            })
            if use:
                block["n3"] = nn.LayerNorm(dim)
                block["lq"] = nn.Linear(dim, dim, bias=False)
                block["lo"] = nn.Linear(dim, dim, bias=False)
            self.blk.append(block)
        self.nf = nn.LayerNorm(dim)

    def forward(self, x, neighbors, pattern_ids):
        batch, width = x.shape
        h = self.tok(x) + self.pos(torch.arange(width, device=x.device))[None] + self.pat(pattern_ids)[:, None, :]
        for block in self.blk:
            q, k, v = block["qkv"](block["n1"](h)).chunk(3, -1)
            q = q.view(batch, width, self.H, -1).transpose(1, 2)
            k = k.view(batch, width, self.H, -1).transpose(1, 2)
            v = v.view(batch, width, self.H, -1).transpose(1, 2)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + block["out"](att.transpose(1, 2).contiguous().view(batch, width, self.d))
            h = h + block["ff2"](F.gelu(block["ff1"](block["n2"](h))))
            if self.use:
                memory = self.tok(neighbors)
                q2 = block["lq"](block["n3"](h)).view(batch, width, self.H, -1)
                k2 = memory.view(batch, width, self.K, self.H, -1).permute(0, 1, 3, 2, 4)
                scores = (q2.unsqueeze(3) @ k2.transpose(-1, -2)) / (self.d // self.H) ** 0.5
                h = h + block["lo"]((scores.softmax(-1) @ k2).squeeze(3).reshape(batch, width, self.d))
        return self.nf(h)


def sentence_tokens(text):
    # Match pack_banao.py exactly: tokenize each input line with the training regex.
    for line in text.splitlines():
        tokens = re.findall(r"[^\s।,?!]+", line)
        if 4 <= len(tokens) <= 25:
            yield tokens


def main():
    if not os.path.isfile(a.ckpt) or not os.path.isfile(a.pack) or not os.path.isfile(a.test_parquet):
        raise SystemExit("Missing checkpoint, training pack, or held-out parquet")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(ck, dict) or "m" not in ck or "vocab" not in ck or "pats" not in ck:
        raise SystemExit("Checkpoint metadata is incomplete")

    seq = int(ck.get("seq", 24))
    pattern_window = int(ck.get("pattern_window", max(1, min(seq, 8))))
    vocab = list(ck["vocab"])
    vocab_size = len(vocab)
    if vocab_size != int(ck.get("V", vocab_size)):
        raise SystemExit("Checkpoint vocab size mismatch")
    word_to_id = {word: i for i, word in enumerate(vocab)}
    pats = [tuple(p) for p in ck["pats"]]
    pattern_to_id = {p: i for i, p in enumerate(pats)}
    stop_ids = set()
    for p in pats:
        stop_ids.update(x for x in p if isinstance(x, int))
    def pattern(prefix):
        return tuple(token_id if token_id in stop_ids else "*" for token_id in prefix[-pattern_window:])

    pack = Pack(a.pack)
    words = pack.words()
    paths = []
    for i in range(min(pack.ns, 200000)):
        q = pack.path(i)
        if q and 6 <= len(q) <= 60:
            paths.append(q)
    if not paths:
        raise SystemExit("Training pack has no usable paths")

    # Recreate sampurna2.py's exact training-time ID remap and candidate tables.
    old_counts = Counter(token for path in paths for token in path)
    top = [token for token, _ in old_counts.most_common(pack.V)]
    remap = {old: new for new, old in enumerate(top)}
    paths = [[remap[token] for token in path if token in remap] for path in paths]
    paths = [path for path in paths if len(path) >= 6]
    if len(top) != vocab_size or [words[i] for i in top] != vocab:
        raise SystemExit("Training pack token map does not match checkpoint vocabulary")

    # Match the original training code exactly (including its original-ID STOP rule).
    stop_original_ids = {token for token, count in old_counts.items() if count >= 300}
    def train_pattern(prefix):
        suffix = prefix[-pattern_window:]
        return tuple(token_id if token_id in stop_original_ids else "*" for token_id in suffix)

    neighbors, graph_candidates = {}, {}
    for old, new in remap.items():
        linked = [remap[x] for x in pack.links(old) if x in remap]
        if linked:
            neighbors[new] = linked[:a.nbr]
            graph_candidates[new] = linked[:a.cand]

    pattern_slots = defaultdict(lambda: defaultdict(Counter))
    pattern_counts = Counter()
    for path in paths:
        for i in range(2, len(path)):
            key = train_pattern(path[:i])
            pattern_counts[key] += 1
            pattern_slots[key][min(i, seq)][path[i]] += 1
    rebuilt_patterns = [key for key, count in pattern_counts.items() if count >= a.minpat]
    if rebuilt_patterns != pats:
        raise SystemExit("Rebuilt pattern table differs from checkpoint metadata; refusing invalid score")
    pattern_candidates = {}
    for key in pats:
        for position, counts in pattern_slots[key].items():
            pattern_candidates[(key, position)] = [
                token for token, _ in counts.most_common(a.pat_cand)
                if 0 <= token < vocab_size
            ]

    use = int(ck.get("use", 1))
    model = Model(vocab_size, len(pats), int(ck["dim"]), int(ck["layers"]),
                  int(ck["heads"]), a.nbr, use)
    model.load_state_dict(ck["m"], strict=True)
    model.eval()
    checkpoint_step = int(ck.get("step", 0))
    del ck

    reservoir = []
    seen_questions = 0
    total_tokens = 0
    known_tokens = 0
    total_sentences = 0
    usable_sentences = 0
    parquet = pq.ParquetFile(a.test_parquet)
    for batch in parquet.iter_batches(batch_size=1000, columns=["text"]):
        for text_value in batch.column(0).to_pylist():
            if not text_value:
                continue
            for tokens in sentence_tokens(text_value):
                total_sentences += 1
                total_tokens += len(tokens)
                if not all(token in word_to_id for token in tokens):
                    known_tokens += sum(token in word_to_id for token in tokens)
                    continue
                known_tokens += len(tokens)
                ids = [word_to_id[token] for token in tokens]
                usable_sentences += 1
                for i in range(2, len(ids)):
                    context = ids[max(0, i-seq):i]
                    if len(context) < 2:
                        continue
                    seen_questions += 1
                    item = (context, ids[i], min(i, seq))
                    if len(reservoir) < a.nq:
                        reservoir.append(item)
                    else:
                        j = random.randrange(seen_questions)
                        if j < a.nq:
                            reservoir[j] = item
        if seen_questions >= max(a.nq * 30, 15000) and len(reservoir) == a.nq:
            break

    if not reservoir:
        raise SystemExit("No fully in-vocabulary external test examples; accuracy not computed")

    n = len(reservoir)
    covered = top1 = top5 = graph1 = pattern1 = 0
    for context, target, position in reservoir:
        pt = train_pattern(context)
        pid = pattern_to_id.get(pt, 0)
        g = graph_candidates.get(context[-1], [])
        pc = pattern_candidates.get((pt, position), [])
        candidates = list(dict.fromkeys(g + pc))[:a.cand + a.pat_cand]
        if g and g[0] == target:
            graph1 += 1
        if pc and pc[0] == target:
            pattern1 += 1
        if target not in candidates:
            continue
        covered += 1
        x = torch.zeros(1, seq, dtype=torch.long)
        nb = torch.zeros(1, seq, a.nbr, dtype=torch.long)
        x[0, :len(context)] = torch.tensor(context, dtype=torch.long)
        for t, token in enumerate(context):
            row = neighbors.get(token, [])[:a.nbr]
            if row:
                nb[0, t, :len(row)] = torch.tensor(row, dtype=torch.long)
        candidate_tensor = torch.tensor(candidates, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            hidden = model(x, nb, torch.tensor([pid], dtype=torch.long))[:, -1, :]
            scores = (hidden[:, None, :] * model.tok(candidate_tensor)).sum(-1)[0]
        order = scores.argsort(descending=True).tolist()
        ranked = [candidates[k] for k in order[:5]]
        if ranked and ranked[0] == target:
            top1 += 1
        if target in ranked:
            top5 += 1

    sha = hashlib.sha256()
    with open(a.ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    result = {
        "domain": os.environ.get("DOMAIN", "unknown"),
        "protocol": "external_hindi_fineweb2_shard_no_target_injection",
        "test_shard": os.environ.get("TEST_SHARD", "unknown"),
        "training_pack_paths": len(paths),
        "checkpoint_step": checkpoint_step,
        "checkpoint_sha256": sha.hexdigest(),
        "questions_sampled": n,
        "questions_available": seen_questions,
        "test_sentences_total": total_sentences,
        "test_sentences_fully_in_vocab": usable_sentences,
        "test_token_coverage": known_tokens / max(total_tokens, 1),
        "candidate_recall": covered / n,
        "candidate_misses_counted_wrong": n - covered,
        "top1_accuracy_all_questions": top1 / n,
        "top5_accuracy_all_questions": top5 / n,
        "top1_given_target_in_candidates": top1 / max(covered, 1),
        "graph_only_top1_all_questions": graph1 / n,
        "pattern_only_top1_all_questions": pattern1 / n,
        "warning": "External shard is disjoint from this domain's assigned FineWeb training shard; full-vocabulary OOV transitions are excluded and token coverage is reported. No true target is inserted into candidate lists."
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
