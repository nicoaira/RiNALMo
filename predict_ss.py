#!/usr/bin/env python
"""
Batch secondary-structure prediction with RInalmo.

Usage
-----
predict_ss.py --input canonical_transcript_sequences_f1437.tsv
              [--seq-col sequence] [--limit 50]

Required
--------
--input      path to a TSV/CSV containing the sequences.

Optional
--------
--seq-col    name of the column that holds the raw RNA sequence
             (default: "sequence")

--limit      process at most N sequences from the top of the file.
             Useful for timing experiments.

Output
------
For each sequence a single line is emitted:
    <row-idx>\t<predicted-dot-bracket>

At the end, timing statistics are printed to stderr.

Author: 2025-05-20
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import pandas as pd

from train_sec_struct_prediction import SecStructPredictionWrapper
from rinalmo.data.alphabet import Alphabet
from rinalmo.utils.sec_struct import prob_mat_to_sec_struct

# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="Batch secondary-structure prediction with RInalmo")
parser.add_argument("--input", required=True, help="TSV/CSV file with sequences")
parser.add_argument("--seq-col", default="sequence", help="Column that stores the RNA sequence")
parser.add_argument("--limit", type=int, default=None, help="Restrict to the first N sequences")
args = parser.parse_args()

# ──────────────────────────────────────────
# Resolve repo root & environment
# ──────────────────────────────────────────
HERE = Path(__file__).resolve().parent
os.chdir(HERE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────
# 1. Load the model (time it)
# ──────────────────────────────────────────
t0 = time.perf_counter()

model = SecStructPredictionWrapper().to(device).eval()

ckpt_path = HERE / "weights" / "rinalmo_giga_ss_bprna_ft.pt"
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state, strict=False)

t_model = time.perf_counter() - t0
print(f"▶ Loaded model, threshold = {model.threshold:.3f}  (t_load = {t_model:.2f}s)", file=sys.stderr)

# ──────────────────────────────────────────
# 2. Prepare Alphabet
# ──────────────────────────────────────────
alphabet = Alphabet(**model.lm.config["alphabet"])

# ──────────────────────────────────────────
# 3. Read sequences
# ──────────────────────────────────────────
df = pd.read_csv(args.input, sep=None, engine="python")  # autodetect TSV / CSV
if args.seq_col not in df.columns:
    sys.exit(f"ERROR: column '{args.seq_col}' not present in {args.input}")

seqs = df[args.seq_col].astype(str).tolist()
if args.limit:
    seqs = seqs[: args.limit]

if not seqs:
    sys.exit("ERROR: no sequences selected")

# ──────────────────────────────────────────
# 4. Inference loop
# ──────────────────────────────────────────
inference_times = []
for idx, seq in enumerate(seqs):
    start = time.perf_counter()

    tokens = torch.tensor(alphabet.batch_tokenize([seq]),
                          dtype=torch.int64,
                          device=device)

    with torch.no_grad(), torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
        logits = model(tokens)[0]          # (L_max, L_max)
        probs  = torch.sigmoid(logits).cpu().numpy()

    L       = len(seq)
    pm_trim = probs[:L, :L]
    pairing = prob_mat_to_sec_struct(pm_trim, seq, model.threshold)

    # Build dot-bracket
    pred_db = ["." for _ in range(L)]
    for i in range(L):
        for j in range(i + 1, L):
            if pairing[i, j] == 1:
                pred_db[i], pred_db[j] = "(", ")"
    pred_db = "".join(pred_db)

    # Emit result to STDOUT
    print(f"{idx}\t{pred_db}")

    inference_times.append(time.perf_counter() - start)

# ──────────────────────────────────────────
# 5. Timing summary
# ──────────────────────────────────────────
total_inf = sum(inference_times)
print(f"▶ Processed {len(seqs)} seqs  (Σ inf = {total_inf:.2f}s, "
      f"μ per-seq = {total_inf/len(seqs):.3f}s)", file=sys.stderr)

