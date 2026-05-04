"""
Modal app for the 4-loss KD comparison + spec_eval_rejection.

Pipeline:
  build_data (~1 min)  -> writes train.pt/val.pt to /vol
  train.map([ce,fkl,rkl,gkd]) in parallel on A100s -> /vol/checkpoints/student_*
  spec_eval -> writes /vol/results/spec_eval_rejection.json

Run:
    modal run modal_kd_app.py
"""

from __future__ import annotations
import modal


CUDA_TAG = "12.4.1-cudnn-devel-ubuntu22.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("git", "wget")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.43",
        "datasets==3.6.0",
        "accelerate>=0.34",
        "tqdm>=4.66",
        "numpy>=1.26",
        "huggingface_hub>=0.24",
    )
    .run_commands(
        "mkdir -p /models",
        "hf download Qwen/Qwen2.5-Coder-0.5B --local-dir /models/qwen-coder-0.5b",
        "hf download Qwen/Qwen2.5-Coder-1.5B --local-dir /models/qwen-coder-1.5b",
    )
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=[".venv", ".git", "checkpoints", "results", "data/cache", "*.pyc", "__pycache__"],
    )
)

volume = modal.Volume.from_name("kd-rejection-results", create_if_missing=True)
app = modal.App("kd-rejection")

QWEN_05B = "/models/qwen-coder-0.5b"
QWEN_15B = "/models/qwen-coder-1.5b"


@app.function(
    image=image,
    cpu=4,
    memory=16 * 1024,
    timeout=20 * 60,
    volumes={"/vol": volume},
)
def build_data(n_train: int = 2048, n_val: int = 128, seq_len: int = 512) -> str:
    """Tokenise codeparrot-clean-valid into train.pt/val.pt on /vol/data."""
    import sys, os
    sys.path.insert(0, "/workspace")
    from pathlib import Path
    out_dir = Path("/vol/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "train.pt").exists() and (out_dir / "val.pt").exists():
        print(f"[build_data] cache already present at {out_dir}, skipping")
        return str(out_dir)

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN_05B)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # streaming so we don't pull the whole 50GB shard
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    ds = ds.shuffle(seed=0, buffer_size=10000)

    train_ids: list[torch.Tensor] = []
    val_ids: list[torch.Tensor] = []
    target_total = n_train + n_val
    n_seen = 0
    for ex in ds:
        text = ex["content"]
        if len(text) < 200:
            continue
        ids = tok(text, return_tensors="pt", truncation=True, max_length=seq_len)["input_ids"][0]
        if ids.size(0) < seq_len // 2:
            continue
        if ids.size(0) < seq_len:
            pad = torch.full((seq_len - ids.size(0),), tok.pad_token_id, dtype=ids.dtype)
            ids = torch.cat([ids, pad])
        n_seen += 1
        if n_seen <= n_train:
            train_ids.append(ids)
        else:
            val_ids.append(ids)
        if n_seen >= target_total:
            break

    train = torch.stack(train_ids)
    val = torch.stack(val_ids)
    torch.save(train, out_dir / "train.pt")
    torch.save(val, out_dir / "val.pt")
    volume.commit()
    print(f"[build_data] train {tuple(train.shape)}  val {tuple(val.shape)} -> {out_dir}")
    return str(out_dir)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=32 * 1024,
    timeout=4 * 60 * 60,
    volumes={"/vol": volume},
)
def train_one(loss: str, steps: int = 2500) -> str:
    """Run distill.py for one loss variant. Returns checkpoint dir on volume."""
    import sys, subprocess, time
    sys.path.insert(0, "/workspace")
    from pathlib import Path

    ckpt_dir = Path("/vol/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / f"student_{loss}"
    if target.exists() and (target / "config.json").exists():
        print(f"[train_one:{loss}] checkpoint already exists, skipping")
        return str(target)

    cmd = [
        "python", "-u", "/workspace/distill.py",
        "--loss", loss,
        "--teacher", QWEN_15B,
        "--student", QWEN_05B,
        "--train-pt", "/vol/data/train.pt",
        "--val-pt", "/vol/data/val.pt",
        "--out-dir", str(ckpt_dir),
        "--steps", str(steps),
        "--eval-every", "500",
        "--checkpoint-every", "250",
        "--batch-size", "4",
        "--grad-accum", "2",
        "--lr", "2e-5",
        "--temperature", "1.0",
        "--seed", "0",
    ]
    print(f"[train_one:{loss}] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd="/workspace")
    # Volume snapshots only persist across preemption when we commit.
    # Commit every 5 min so a worker restart loses at most one window of work.
    last_commit = time.time()
    while proc.poll() is None:
        time.sleep(15)
        if time.time() - last_commit > 300:
            volume.commit()
            last_commit = time.time()
            print(f"[train_one:{loss}] volume.commit()", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"distill.py exited {proc.returncode}")
    volume.commit()
    return str(target)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=32 * 1024,
    timeout=2 * 60 * 60,
    volumes={"/vol": volume},
)
def spec_eval() -> str:
    """Run spec_eval_rejection.py against the 4 trained students."""
    import sys, subprocess
    sys.path.insert(0, "/workspace")
    from pathlib import Path

    out_path = Path("/vol/results/spec_eval_rejection.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-u", "/workspace/spec_eval_rejection.py",
        "--teacher", QWEN_15B,
        "--student-base", QWEN_05B,
        "--ckpt-dir", "/vol/checkpoints",
        "--out", str(out_path),
        "--K", "4",
        "--max-drafts", "8",
        "--eval-seed", "42",
    ]
    print(f"[spec_eval] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd="/workspace")
    volume.commit()
    return str(out_path)


@app.local_entrypoint()
def main():
    print("== build_data ==")
    build_data.remote()

    print("== train (4 losses in parallel) ==")
    losses = ["ce", "fkl", "rkl", "gkd"]
    handles = [train_one.spawn(L) for L in losses]
    for L, h in zip(losses, handles):
        result = h.get()
        print(f"  {L} -> {result}")

    print("== spec_eval ==")
    out = spec_eval.remote()
    print(f"results -> {out}")
    print("done")
