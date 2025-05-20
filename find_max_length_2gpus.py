#!/usr/bin/env python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"

from pathlib import Path
import torch
from train_sec_struct_prediction import SecStructPredictionWrapper

# ────────────────────────────────────────────────────────────────
# 1) cd into the repo root so imports & paths resolve
HERE = Path(__file__).resolve().parent
os.chdir(HERE)

# 2) Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SecStructPredictionWrapper().to(device).eval()

# 3) Load the full checkpoint (threshold + lm + head)
ckpt = torch.load(HERE/"weights"/"rinalmo_giga_ss_bprna_ft.pt", map_location=device)
model.load_state_dict(ckpt, strict=False)

def fits_nt(num_nt: int) -> bool:
    """
    Returns True if a dummy RNA of num_nt nucleotides (num_nt+2 tokens) can be
    processed end‐to‐end (LM + SS head) without CUDA OOM or allocator errors.
    """
    tokens_len = num_nt + 2
    dummy = torch.zeros((1, tokens_len), dtype=torch.int64, device=device)
    torch.cuda.empty_cache()
    try:
        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            _ = model(dummy)
        return True
    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg or "INTERNAL ASSERT FAILED" in msg:
            # treat allocator assertion as a fit‐failure
            torch.cuda.empty_cache()
            return False
        raise
    except ValueError as e:
        msg = str(e)
        if "Expected more than 1 spatial element" in msg:
            # tiny spatial-dim error only; ignore
            torch.cuda.empty_cache()
            return True
        raise
    finally:
        torch.cuda.empty_cache()

# ────────────────────────────────────────────────────────────────
print("Finding max RNA length (nucleotides) that fits end-to-end…")

# 1) Exponential bracketing to find an upper bound
low, high = 1, 1
while fits_nt(high):
    low = high
    high *= 2
    if high > 5000:
        break
print(f"Bracketed between {low} nts (OK) and {high} nts (OOM or cap)")

# 2) Binary search within [low, high)
best = low
while low + 1 < high:
    mid = (low + high) // 2
    if fits_nt(mid):
        best = mid
        low = mid
    else:
        high = mid

print(f"\n✅ Maximum RNA length that fits (nucleotides): {best} nts")
