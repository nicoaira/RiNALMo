#!/usr/bin/env python
import os
from pathlib import Path

import torch
from train_sec_struct_prediction import SecStructPredictionWrapper
from rinalmo.data.alphabet import Alphabet
from rinalmo.utils.sec_struct import prob_mat_to_sec_struct

# ─────────────────────────────────────────────────────────────────────────────────────
# 1) cd into the repo root so imports & relative paths resolve
HERE = Path(__file__).resolve().parent
os.chdir(HERE)

# 2) Instantiate wrapper exactly as in training (no overrides)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SecStructPredictionWrapper().to(device).eval()

# 3) Load the full checkpoint (contains "threshold", "lm.*", "pred_head.*")
ckpt_path = HERE / "weights" / "rinalmo_giga_ss_bprna_ft.pt"
state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state, strict=False)
print(f"▶ Loaded threshold = {model.threshold:.3f}")

# 4) Prepare the Alphabet
alphabet = Alphabet(**model.lm.config["alphabet"])

# 5) Single test sequence\

seq = (
    "GUUGAGGUGGAUAAUAGGCACGUGAACACGAGCGUAUUUCAUGAUAUAGCUCGGUAGGUUGGAACGGCACGUCUUCGCACCCCGGUCUCGCGAGAUCAUCUUACGUAGUAAAUGGUACUGCCUUGAUACGAGAUGACGACUGCAGUUAUUUUGUGCUCGUCCCUUUAUUCAGGAGGGCCGCCUGACUCCGGCUUCUACCCGAUCAUUCUUGAGGUCCACGGGUGAUACCCUGUAGCUUCGACAGCACAGACUGGAGCAAAAGAUACCUAGUACUCCAAGGCUGAUCGUUUCGCGCAGUGAUGAGCUAAGCGU"
)

# 6) Tokenize & forward
tokens = torch.tensor(alphabet.batch_tokenize([seq]), dtype=torch.int64, device=device)
print("Before forward:\n", torch.cuda.memory_summary(), flush=True)
with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
    logits = model(tokens)[0]          # (L_max, L_max)
    probs  = torch.sigmoid(logits).cpu().numpy()
print("After forward:\n", torch.cuda.memory_summary(), flush=True)

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

# 9) Output the predicted dot-bracket structure
print("predictions")
print(pred_db)
