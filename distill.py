"""
Distill Qwen2.5-Coder-1.5B (teacher) into Qwen2.5-Coder-0.5B (student).

One script, --loss flag picks the variant.

  ce          plain next-token cross-entropy on the corpus, teacher unused.
              Baseline -- if a method beats this, distillation is doing work.
  fkl         forward KL with temperature scaling (Hinton et al., 2015).
  rkl         reverse KL on teacher-forced positions (the off-policy half of
              MiniLLM's recipe; Gu et al., 2024).
  gkd         on-policy reverse KL: sample student rollouts from the prefix,
              compute teacher probs on the same prefixes, minimise reverse KL
              there (Agarwal et al., 2024 -- GKD).

GKD is the cheap stand-in for the full MiniLLM REINFORCE estimator. It
captures the fix for distribution shift without policy-gradient variance.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import math
import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Cfg:
    loss: str
    teacher: str
    student: str
    train_pt: Path
    val_pt: Path
    out_dir: Path
    steps: int = 3000
    eval_every: int = 500
    batch_size: int = 4
    grad_accum: int = 2
    lr: float = 2e-5
    warmup: int = 100
    temperature: float = 1.0
    rollout_temperature: float = 1.0
    rollout_prompt_len: int = 64
    rollout_new: int = 64
    seed: int = 0
    log_every: int = 50
    # Lost a 2150/2500 gkd run to Modal preemption once. Save full state
    # (model, optimizer, scheduler, RNG, step, log) every checkpoint_every
    # steps so the next preemption costs at most one window.
    checkpoint_every: int = 250


def cosine_warmup(opt: torch.optim.Optimizer, warmup: int, total: int) -> LambdaLR:
    def fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(opt, lr_lambda=fn)


def shift_logits_targets(
    logits: torch.Tensor, ids: torch.Tensor, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (logits[:-1], targets[1:], mask) -- mask is 1 where target != pad."""
    sl = logits[:, :-1, :].contiguous()
    tg = ids[:, 1:].contiguous()
    mask = (tg != pad_id).float()
    return sl, tg, mask


def ce_loss(s_logits: torch.Tensor, ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    sl, tg, mask = shift_logits_targets(s_logits, ids, pad_id)
    flat = sl.view(-1, sl.size(-1))
    tgt = tg.view(-1)
    losses = F.cross_entropy(flat, tgt, reduction="none").view_as(tg)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def forward_kl(
    s_logits: torch.Tensor, t_logits: torch.Tensor, mask: torch.Tensor, T: float
) -> torch.Tensor:
    """KL(p_T || p_S) per token, averaged over non-pad positions."""
    log_s = F.log_softmax(s_logits / T, dim=-1)
    p_t = F.softmax(t_logits / T, dim=-1)
    log_t = F.log_softmax(t_logits / T, dim=-1)
    per_tok = (p_t * (log_t - log_s)).sum(dim=-1)
    return (T * T) * (per_tok * mask).sum() / mask.sum().clamp_min(1.0)


def reverse_kl(
    s_logits: torch.Tensor, t_logits: torch.Tensor, mask: torch.Tensor, T: float
) -> torch.Tensor:
    """KL(p_S || p_T) per token, averaged over non-pad positions."""
    log_s = F.log_softmax(s_logits / T, dim=-1)
    log_t = F.log_softmax(t_logits / T, dim=-1)
    p_s = log_s.exp()
    per_tok = (p_s * (log_s - log_t)).sum(dim=-1)
    return (T * T) * (per_tok * mask).sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def sample_student_rollouts(
    student: torch.nn.Module,
    prompt_ids: torch.Tensor,
    n_new: int,
    temperature: float,
) -> torch.Tensor:
    student.eval()
    out = student.generate(
        prompt_ids,
        max_new_tokens=n_new,
        do_sample=True,
        temperature=temperature,
        top_k=0,
        top_p=1.0,
        pad_token_id=student.config.eos_token_id,
    )
    student.train()
    return out


@torch.no_grad()
def eval_perplexity(
    model: torch.nn.Module, val_ids: torch.Tensor, pad_id: int, batch: int = 4
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0.0
    for i in range(0, val_ids.size(0), batch):
        b = val_ids[i : i + batch].cuda()
        out = model(b)
        sl, tg, mask = shift_logits_targets(out.logits, b, pad_id)
        flat = sl.view(-1, sl.size(-1))
        tgt = tg.view(-1)
        losses = F.cross_entropy(flat, tgt, reduction="none").view_as(tg)
        total_loss += (losses * mask).sum().item()
        total_tokens += mask.sum().item()
    model.train()
    return total_loss / max(total_tokens, 1.0)


def _partial_dir(cfg: Cfg) -> Path:
    return cfg.out_dir / f"student_{cfg.loss}_partial"


def _save_checkpoint(
    cfg: Cfg, step: int, student, opt, sched, rng, log: dict, t_offset: float
) -> None:
    d = _partial_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(d / "model")
    torch.save({
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "rng": rng.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }, d / "trainer.pt")
    (d / "meta.json").write_text(json.dumps({
        "step": step,
        "t_offset": t_offset,
        "log": log,
    }))


def _try_resume(cfg: Cfg) -> dict | None:
    d = _partial_dir(cfg)
    if not (d / "meta.json").exists() or not (d / "trainer.pt").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    print(f"[resume] partial checkpoint at step {meta['step']}, loading")
    return meta


def distill(cfg: Cfg) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(cfg.student)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    resumed = _try_resume(cfg)
    student_src = (
        str(_partial_dir(cfg) / "model") if resumed is not None else cfg.student
    )
    student = AutoModelForCausalLM.from_pretrained(
        student_src, torch_dtype=torch.bfloat16
    ).to(device)
    student.gradient_checkpointing_enable()
    student.train()

    teacher = None
    if cfg.loss != "ce":
        teacher = AutoModelForCausalLM.from_pretrained(
            cfg.teacher, torch_dtype=torch.bfloat16
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    train_ids = torch.load(cfg.train_pt)
    val_ids = torch.load(cfg.val_pt)
    print(
        f"loss={cfg.loss}  T={cfg.temperature}  steps={cfg.steps}  "
        f"effective_bs={cfg.batch_size * cfg.grad_accum}  "
        f"train={tuple(train_ids.shape)}  val={tuple(val_ids.shape)}"
    )
    print(f"student params: {sum(p.numel() for p in student.parameters()):,}")
    if teacher is not None:
        print(f"teacher params: {sum(p.numel() for p in teacher.parameters()):,}")

    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = cosine_warmup(opt, cfg.warmup, cfg.steps)

    rng = torch.Generator(device="cpu").manual_seed(cfg.seed)
    n_train = train_ids.size(0)

    log: dict = {"step": [], "train_loss": [], "val_nll": [], "wallclock_s": []}
    t_offset = 0.0
    start_step = 1
    if resumed is not None:
        trainer = torch.load(_partial_dir(cfg) / "trainer.pt")
        opt.load_state_dict(trainer["optimizer"])
        sched.load_state_dict(trainer["scheduler"])
        rng.set_state(trainer["rng"])
        torch.set_rng_state(trainer["torch_rng"])
        if trainer["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(trainer["cuda_rng"])
        log = resumed["log"]
        t_offset = float(resumed["t_offset"])
        start_step = int(resumed["step"]) + 1
        print(f"[resume] starting at step {start_step}, t_offset={t_offset:.0f}s")

    t0 = time.time() - t_offset
    if resumed is None:
        val0 = eval_perplexity(student, val_ids, pad_id, batch=cfg.batch_size)
        print(f"step 0  val_nll={val0:.4f}")
        log["step"].append(0)
        log["train_loss"].append(None)
        log["val_nll"].append(val0)
        log["wallclock_s"].append(0.0)

    accum = 0
    accum_loss = 0.0
    opt.zero_grad(set_to_none=True)
    for step in range(start_step, cfg.steps + 1):
        idx = torch.randint(0, n_train, (cfg.batch_size,), generator=rng)
        batch = train_ids[idx].to(device, non_blocking=True)

        if cfg.loss == "ce":
            out = student(batch)
            loss = ce_loss(out.logits, batch, pad_id)
        elif cfg.loss in ("fkl", "rkl"):
            with torch.no_grad():
                t_logits = teacher(batch).logits
            s_out = student(batch)
            sl, _, mask = shift_logits_targets(s_out.logits, batch, pad_id)
            tl = t_logits[:, :-1, :].contiguous()
            if cfg.loss == "fkl":
                loss = forward_kl(sl, tl, mask, cfg.temperature)
            else:
                loss = reverse_kl(sl, tl, mask, cfg.temperature)
        elif cfg.loss == "gkd":
            prompt = batch[:, : cfg.rollout_prompt_len]
            full = sample_student_rollouts(
                student, prompt, n_new=cfg.rollout_new,
                temperature=cfg.rollout_temperature,
            )
            with torch.no_grad():
                t_logits = teacher(full).logits
            s_out = student(full)
            sl, _, mask = shift_logits_targets(s_out.logits, full, pad_id)
            tl = t_logits[:, :-1, :].contiguous()
            # only score the rollout positions, not the original prompt
            mask = mask.clone()
            mask[:, : cfg.rollout_prompt_len - 1] = 0
            loss = reverse_kl(sl, tl, mask, cfg.temperature)
        else:
            raise ValueError(cfg.loss)

        loss = loss / cfg.grad_accum
        loss.backward()
        accum_loss += loss.item()
        accum += 1

        if accum >= cfg.grad_accum:
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            accum = 0
            full_loss = accum_loss
            accum_loss = 0.0
        else:
            full_loss = None
        # step the lr scheduler every iteration so the cosine schedule
        # decays over real wall-clock training, not over optimiser steps
        sched.step()

        if step % cfg.log_every == 0:
            wc = time.time() - t0
            shown = full_loss if full_loss is not None else loss.item() * cfg.grad_accum
            print(
                f"step {step:5d}/{cfg.steps}  loss={shown:.4f}  "
                f"lr={sched.get_last_lr()[0]:.2e}  ({wc:.0f}s)"
            )

        if step % cfg.eval_every == 0 or step == cfg.steps:
            v = eval_perplexity(student, val_ids, pad_id, batch=cfg.batch_size)
            wc = time.time() - t0
            print(f"step {step:5d}  val_nll={v:.4f}  ({wc:.0f}s)")
            log["step"].append(step)
            log["train_loss"].append(full_loss)
            log["val_nll"].append(v)
            log["wallclock_s"].append(wc)

        if cfg.checkpoint_every > 0 and step % cfg.checkpoint_every == 0 and step != cfg.steps:
            _save_checkpoint(cfg, step, student, opt, sched, rng, log, t_offset=time.time() - t0)
            print(f"[ckpt] step {step} -> {_partial_dir(cfg)}", flush=True)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(cfg.out_dir / f"student_{cfg.loss}")
    log_path = Path("results") / f"train_{cfg.loss}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"cfg": {**asdict(cfg), "train_pt": str(cfg.train_pt), "val_pt": str(cfg.val_pt), "out_dir": str(cfg.out_dir)}, "log": log}, indent=2, default=str))
    print(f"saved log -> {log_path}")
    print(f"saved student -> {cfg.out_dir / f'student_{cfg.loss}'}")

    # final checkpoint exists, drop the partial dir
    pd = _partial_dir(cfg)
    if pd.exists():
        import shutil
        shutil.rmtree(pd, ignore_errors=True)
        print(f"[ckpt] cleared {pd}")
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", required=True, choices=["ce", "fkl", "rkl", "gkd"])
    ap.add_argument("--teacher", default="/home/prannayk/models/qwen-coder-1.5b")
    ap.add_argument("--student", default="/home/prannayk/models/qwen-coder-0.5b")
    ap.add_argument("--train-pt", type=Path, default=Path("data/cache/train.pt"))
    ap.add_argument("--val-pt", type=Path, default=Path("data/cache/val.pt"))
    ap.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--checkpoint-every", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Cfg(
        loss=args.loss,
        teacher=args.teacher,
        student=args.student,
        train_pt=args.train_pt,
        val_pt=args.val_pt,
        out_dir=args.out_dir,
        steps=args.steps,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        temperature=args.temperature,
        seed=args.seed,
    )
    distill(cfg)


if __name__ == "__main__":
    main()
