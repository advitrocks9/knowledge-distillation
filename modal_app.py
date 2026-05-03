"""
Modal app for the Mellum-as-teacher FIM seq-KD experiment.

Single-function pipeline:
    build FIM data ->
    generate Mellum middles ->
    train (gold | mellum | mix) ->
    eval on held-out FIM exact-match + HumanEval Infilling

Models are baked into the image so the function start-up is fast on
each invocation. Results are written to a persistent volume that I
pull back locally with `modal volume get`.

GPU is A10G (24GB), enough for Mellum-4b BF16 inference + Qwen-0.5B
training. ~3-4h on this size.
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
        # 3.6 is the last datasets release that supports loader scripts;
        # loubnabnl/humaneval_infilling is a script-based dataset and the
        # parquet alternatives don't preserve Bavarian et al.'s original masks
        "datasets==3.6.0",
        "accelerate>=0.34",
        "tqdm>=4.66",
        "numpy>=1.26",
        "huggingface_hub>=0.24",
    )
    .run_commands(
        # download the three models once at image build, cached in the layer
        "mkdir -p /models",
        "hf download Qwen/Qwen2.5-Coder-0.5B --local-dir /models/qwen-coder-0.5b",
        "hf download Qwen/Qwen2.5-Coder-1.5B --local-dir /models/qwen-coder-1.5b",
        "hf download JetBrains/Mellum-4b-sft-python --local-dir /models/mellum-sft-python",
    )
    .add_local_dir(".", remote_path="/workspace", ignore=[".venv", ".git", "checkpoints", "results", "data/cache", "*.pyc", "__pycache__"])
)

volume = modal.Volume.from_name("kd-fim-results", create_if_missing=True)
app = modal.App("kd-fim")

QWEN_05B = "/models/qwen-coder-0.5b"
QWEN_15B = "/models/qwen-coder-1.5b"
MELLUM   = "/models/mellum-sft-python"


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/results_vol": volume},
    timeout=4 * 3600,
)
def run_pipeline(
    skip_data: bool = False,
    skip_generate: bool = False,
    skip_train: bool = False,
    skip_fim_eval: bool = False,
    skip_humaneval: bool = False,
) -> dict:
    """End-to-end FIM seq-KD pipeline."""
    import subprocess
    import sys
    import shutil
    from pathlib import Path
    import torch

    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}, "
          f"device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, "
          f"free MiB {torch.cuda.mem_get_info()[0] // 1024**2 if torch.cuda.is_available() else 0}")

    workdir = Path("/workspace")
    Path("/results_vol").mkdir(exist_ok=True)
    Path("/workspace/data/fim").mkdir(parents=True, exist_ok=True)
    Path("/workspace/results").mkdir(parents=True, exist_ok=True)
    Path("/workspace/checkpoints").mkdir(parents=True, exist_ok=True)

    # rehydrate from volume if we're skipping early stages
    vol_data = Path("/results_vol/data_fim")
    if vol_data.exists():
        print(f"rehydrating data/fim from volume ({sum(1 for _ in vol_data.iterdir())} files)")
        for p in vol_data.iterdir():
            shutil.copy(p, Path("/workspace/data/fim") / p.name)
    vol_ckpt = Path("/results_vol/checkpoints")
    if vol_ckpt.exists():
        for sub in vol_ckpt.iterdir():
            if sub.is_dir():
                dst = Path("/workspace/checkpoints") / sub.name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(sub, dst)
                print(f"rehydrated checkpoint {sub.name} from volume")

    def run(cmd: list[str], stage: str) -> None:
        print(f"\n=== {stage} ===")
        print("$ " + " ".join(cmd))
        out = subprocess.run(cmd, cwd=str(workdir), check=True)

    # 1. build train + eval FIM examples
    if not skip_data:
        run([sys.executable, "fim_data.py",
             "--n-per-kind", "200",
             "--skip", "10000",
             "--seed", "0",
             "--out", "data/fim/train_examples.jsonl"], "fim_data train")
        run([sys.executable, "fim_data.py",
             "--n-per-kind", "60",
             "--skip", "18000",
             "--seed", "1",
             "--out", "data/fim/eval_examples.jsonl"], "fim_data eval")

    # 2. mellum generates the middles for the seq-KD condition
    if not skip_generate:
        run([sys.executable, "fim_generate.py",
             "--teacher", MELLUM,
             "--in-jsonl", "data/fim/train_examples.jsonl",
             "--out-jsonl", "data/fim/mellum_completions.jsonl",
             "--max-new", "128",
             "--batch-size", "8",
             "--greedy"], "mellum generate")
        save_to_volume()  # generation is the slowest step, persist its output

    # 3. train three students
    if not skip_train:
        for src in ("gold", "mellum", "mix"):
            run([sys.executable, "fim_train.py",
                 "--middle", src,
                 "--student", QWEN_05B,
                 "--data", "data/fim/mellum_completions.jsonl",
                 "--steps", "1200",
                 "--lr", "2e-5",
                 "--eval-every", "300"], f"fim train {src}")
            save_to_volume()

    # 4. held-out FIM exact-match eval
    if not skip_fim_eval:
        run([sys.executable, "fim_eval.py",
             "--student-base", QWEN_05B,
             "--data", "data/fim/eval_examples.jsonl",
             "--limit", "180"], "fim eval")
        save_to_volume()

    # 5. HumanEval Infilling -- the published Mellum benchmark
    if not skip_humaneval:
        run([sys.executable, "humaneval_infilling.py",
             "--student-base", QWEN_05B,
             "--mellum", MELLUM,
             "--n-problems", "164",
             "--max-new", "256"], "humaneval infilling")

    save_to_volume()
    print("\n=== done. results saved to volume kd-fim-results ===")
    return {"status": "ok"}


def save_to_volume() -> None:
    """Copy results and FIM data to the persistent volume.
    Called multiple times in the pipeline so partial progress survives if
    the pipeline is interrupted."""
    import shutil
    from pathlib import Path
    src_results = Path("/workspace/results")
    dst_results = Path("/results_vol/results")
    if src_results.exists():
        if dst_results.exists():
            shutil.rmtree(dst_results)
        shutil.copytree(src_results, dst_results)
    src_data = Path("/workspace/data/fim")
    dst_data = Path("/results_vol/data_fim")
    if src_data.exists():
        if dst_data.exists():
            shutil.rmtree(dst_data)
        shutil.copytree(src_data, dst_data)
    src_ck = Path("/workspace/checkpoints")
    dst_ck = Path("/results_vol/checkpoints")
    if src_ck.exists():
        if dst_ck.exists():
            shutil.rmtree(dst_ck)
        shutil.copytree(src_ck, dst_ck)


@app.local_entrypoint()
def main(
    skip_data: bool = False,
    skip_generate: bool = False,
    skip_train: bool = False,
    skip_fim_eval: bool = False,
    skip_humaneval: bool = False,
) -> None:
    fc = run_pipeline.spawn(
        skip_data=skip_data,
        skip_generate=skip_generate,
        skip_train=skip_train,
        skip_fim_eval=skip_fim_eval,
        skip_humaneval=skip_humaneval,
    )
    print(f"spawned function call: {fc.object_id}")
    print("function continues running on Modal even if this client exits")
    print("`modal app logs <app-id>` to follow")
