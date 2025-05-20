#!/usr/bin/env python
import os
from pathlib import Path

import torch
from train_sec_struct_prediction import SecStructPredictionWrapper
from rinalmo.data.alphabet import Alphabet
from rinalmo.utils.sec_struct import prob_mat_to_sec_struct

def pairs_from_dotbracket(db: str):
    stack = []
    pairs = set()
    for idx, c in enumerate(db):
        if c == "(":
            stack.append(idx)
        elif c == ")":
            i = stack.pop()
            pairs.add((i, idx))
    return pairs

def compute_bp_f1(pred_db: str, true_db: str):
    pred = pairs_from_dotbracket(pred_db)
    true = pairs_from_dotbracket(true_db)
    tp = pred & true
    fp = pred - true
    fn = true - pred
    prec = len(tp) / len(pred) if pred else 0.0
    rec  = len(tp) / len(true) if true else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return prec, rec, f1, sorted(fp), sorted(fn)

# ─────────────────────────────────────────────────────────────────────────────────────
# 1) cd into the repo root so imports & relative paths resolve
HERE = Path(__file__).resolve().parent
os.chdir(HERE)

# 2) Instantiate wrapper exactly as in training (no overrides)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SecStructPredictionWrapper()
model = model.to(device).eval()

# 3) Load the full checkpoint (contains "threshold", "lm.*", "pred_head.*")
ckpt_path = HERE / "weights" / "rinalmo_giga_ss_bprna_ft.pt"
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state, strict=False)
print(f"▶ Loaded threshold = {model.threshold:.3f}")

# 4) Prepare the Alphabet
alphabet = Alphabet(**model.lm.config["alphabet"])

# 5) Single test sequence + its true dot-bracket
seq     = "GUCAGGAUGGCCGAGUGGUCUAAGGCGCCAGACUCAAGUUCUGGUCUCCGGAUGGAGGCGUGGGUUCGAAUCCCACUUCUGACA"
true_db = "(((((((..(((...........))).(((((.......))))).((((....))))..(((((.......))))))))))))."

# 6) Tokenize & forward
tokens = torch.tensor(alphabet.batch_tokenize([seq]), dtype=torch.int64, device=device)
with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
    logits = model(tokens)[0]          # (L_max, L_max)
    probs  = torch.sigmoid(logits).cpu().numpy()

# 7) Crop to actual length and run the same masking/greedy cleaning as in train_sec_struct_prediction
L       = len(seq)
pm_trim = probs[:L, :L]
pairing = prob_mat_to_sec_struct(pm_trim, seq, model.threshold)

# 8) Build predicted dot-bracket
pred_db = ["." for _ in range(L)]
for i in range(L):
    for j in range(i+1, L):
        if pairing[i, j] == 1:
            pred_db[i], pred_db[j] = "(", ")"
pred_db = "".join(pred_db)

# 9) Compute and print F1
prec, rec, f1, fp, fn = compute_bp_f1(pred_db, true_db)

print("\nSequence:  ", seq)
print("True DB:   ", true_db)
print("Pred DB:   ", pred_db)
print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
print("False +:   ", fp)
print("False -:   ", fn)