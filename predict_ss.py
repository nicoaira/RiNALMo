#!/usr/bin/env python
"""
RInalmo secondary-structure prediction (inference only).

Usage
-----
predict_ss.py --input <file.tsv> [--seq-col sequence]

Outputs
-------
<row-idx>\t<dot-bracket>            to STDOUT
"""
import argparse
import os
from pathlib import Path

import torch
import pandas as pd

from train_sec_struct_prediction import SecStructPredictionWrapper
from rinalmo.data.alphabet import Alphabet
from rinalmo.utils.sec_struct import prob_mat_to_sec_struct

# ─────────────────────────────── CLI ────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="TSV/CSV with sequences")
p.add_argument("--seq-col", default="sequence")
args = p.parse_args()

# ─────────────────────── repo root & model load ─────────────────────
HERE = Path(__file__).resolve().parent
os.chdir(HERE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SecStructPredictionWrapper().to(device).eval()
state = torch.load(HERE / "weights" / "rinalmo_giga_ss_bprna_ft.pt",
                   map_location=device)
model.load_state_dict(state, strict=False)

alphabet = Alphabet(**model.lm.config["alphabet"])

# ────────────────────────── read sequences ──────────────────────────
df = pd.read_csv(args.input, sep=None, engine="python")
if args.seq_col not in df.columns:
    raise SystemExit(f"column '{args.seq_col}' not found in {args.input}")

seqs = df[args.seq_col].astype(str).tolist()

# ──────────────────────── inference loop ────────────────────────────
for idx, seq in enumerate(seqs):
    tokens = torch.tensor(alphabet.batch_tokenize([seq]),
                          dtype=torch.int64,
                          device=device)
    with torch.no_grad(), torch.autocast(device_type=device.type):
        logits = model(tokens)[0]
        probs  = torch.sigmoid(logits).cpu().numpy()

    L       = len(seq)
    pm_trim = probs[:L, :L]
    pairing = prob_mat_to_sec_struct(pm_trim, seq, model.threshold)

    dot = ["." for _ in range(L)]
    for i in range(L):
        for j in range(i + 1, L):
            if pairing[i, j]:
                dot[i], dot[j] = "(", ")"

    print(f"{idx}\t{''.join(dot)}")
