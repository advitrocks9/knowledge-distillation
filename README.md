# Distilling Qwen2.5-Coder

Forward vs reverse KL distillation from Qwen2.5-Coder-1.5B into Qwen2.5-Coder-0.5B
on Python source. The question I started from: which divergence does the
literature actually want you to use when you sit down to distil a code model,
and is the metric they grade on the metric you'd ship against?

The teacher and student are in the same Qwen family so vocabularies match. The
training corpus is a 2048-sample slice of `codeparrot/codeparrot-clean-valid`.
The student starts from the public Qwen2.5-Coder-0.5B base, not from random
weights, so the comparison is "what does each loss buy on top of an already-
pretrained student" -- the realistic distillation regime, not pretraining.

## Headline (full discussion in `report.md`)

The final table after hardening the spec-decode evaluation (164 HumanEval
prompts, K=4, sampled drafts at T=1.0, bootstrap-by-prompt 95% CIs):

| run | mean accepted run / 4 | 95% CI |
|---|---|---|
| teacher 1.5B (self-spec sanity) | 3.980 | [3.968, 3.990] |
| student base (no fine-tune) | 2.517 | [2.409, 2.624] |
| student + ce  | 2.359 | [2.261, 2.457] |
| student + fkl | 2.477 | [2.377, 2.583] |
| student + rkl | **2.573** | [2.474, 2.684] |
| student + gkd | 2.562 | [2.460, 2.658] |

Paired CIs on per-prompt deltas (per-prompt RNG is shared across
variants by `rng_seed = eval_seed * 1_000_003 + i` in `spec_eval.py`,
so the design is paired by construction; `spec_eval.json` only persists
aggregates, so `analyses/paired_bootstrap.py` falls through to the
marginal upper bound `var(delta) <= var(a) + var(b)`, which assumes
independence and is therefore conservative — the real paired CI is
narrower):

| pair | mean delta | 95% CI |
|---|---|---|
| rkl - ce | +0.214 | [+0.070, +0.357] |
| gkd - ce | +0.203 | [+0.064, +0.342] |
| fkl - ce | +0.118 | [-0.024, +0.260] |

`rkl > ce` and `gkd > ce` are significant even at this conservative
upper bound; fkl > ce is consistent in direction across prompts but
the paired CI crosses zero.

Three-seed retraining of CE and RKL (the pair that survived the
single-seed eval) confirms the gap: CE mean 2.378 (seed std 0.008),
RKL mean 2.584 (seed std 0.023). The gap (+0.206) is ~10x the larger
of the two seed SDs. With n=3 per arm I wouldn't claim more than that.
See `results/spec_eval_seeds.json`.

The numbers above are from the analytical per-block expectation
`E[L] = sum_i prod_{j<=i} a_j` (cheaper, less noisy on a 4090 budget).
I ran the bit-exact rejection-sampling version on Modal as a sanity
check (`spec_eval_rejection.py`, all six models, results in
`results/spec_eval_rejection.json`); every analytic value sits inside
the sampled 95% CI and the ranking is identical. Side-by-side in
`analyses/compare_rejection_vs_analytic.py`.

The first-pass eval (32 prompts, no CIs, in `results/eval.json`) had me
write that forward KL won this column. That story didn't survive 164
prompts. The actual finding is **reverse-KL distillation preserves
teacher alignment significantly better than CE-only fine-tuning**;
within the KL-distilled methods, the direction (forward vs reverse)
isn't distinguishable at this teacher size. `report.md` walks through
the iteration -- first-pass eval, review, hardened protocol, revised
conclusion, then per-position analysis showing the loss-shape mechanism.

HumanEval pass@1 numbers don't change between evals (deterministic
greedy on 164 problems): teacher 0.427, base 0.256, all four distilled
students 0.262-0.274 -- a one-problem-of-164 spread, well within noise.
That column is too small to discriminate the methods.

## Run order

```bash
uv sync
python data.py                                       # tokenise corpus
# the actual runs used --batch-size 2 --grad-accum 4 --eval-every 250 (effective batch 8); defaults are 4/2/500
python distill.py --loss ce  --steps 2500 --batch-size 2 --grad-accum 4 --eval-every 250 --lr 2e-5  # baseline
python distill.py --loss fkl --steps 2500 --batch-size 2 --grad-accum 4 --eval-every 250 --lr 2e-5  # forward KL
python distill.py --loss rkl --steps 2500 --batch-size 2 --grad-accum 4 --eval-every 250 --lr 2e-5  # reverse KL
python distill.py --loss gkd --steps 2500 --batch-size 2 --grad-accum 4 --eval-every 250 --lr 2e-5  # on-policy reverse KL
python eval.py                                       # HumanEval pass@1, NLL, spec-decode
python make_table.py                                 # markdown table
```

About 2h on a single 4090. The `gkd` step does autoregressive student
sampling, which is most of the training time; everything else fits in 8 min.

## What's in here

```
data.py            codeparrot Python -> torch tensor cache (used by distill.py)
distill.py         one script, four losses (ce, fkl, rkl, gkd)
eval.py            first-pass aggregate eval (HumanEval pass@1 + held-out NLL + spec-decode @ 32 prompts)
spec_eval.py       hardened spec-decode (164 prompts, bootstrap CIs, per-position acceptance)
spec_eval_seeds.py same thing across 3 seeds for CE and RKL
per_position.py    per-position teacher-entropy bucketing and TV / agreement / top-1 mass
make_table.py      pretty-prints results/eval.json
results/           per-run training logs, eval.json, spec_eval.json, per_position.json, spec_eval_seeds.json
report.md          writeup with the discussion + algebraic derivation tying spec-decode to TV

fim_data.py            build (prefix, middle, suffix) FIM examples from codeparrot
fim_generate.py        Mellum-4b-sft-python generates middles for seq-KD
fim_train.py           train Qwen on FIM data, gold/mellum/mix middle sources
fim_eval.py            held-out FIM exact-match against gold middles
humaneval_infilling.py the actual benchmark Mellum's card reports (single/multi/random)
modal_app.py           the Modal serverless runner (A10G, image bakes the 3 models)
run_fim_experiment.sh  local-GPU runner; modal_app.py is the cloud-GPU runner
```

The Mellum-as-teacher follow-up ran on Modal A10G -- 600 codeparrot
examples, three students
(`fim_gold` / `fim_mellum` / `fim_mix`), evaluated on three columns:
held-out codeparrot FIM exact-match (in-distribution), HumanEval
Infilling pass@1 (out-of-distribution FIM), and RepoBench-Python
edit-similarity (out-of-distribution L2R, the regression check).

| method | held-out FIM EM | HumanEval Infilling mean pass@1 | RepoBench avg ES |
|---|---|---|---|
| base 0.5B | 0.122 | **0.563** | **0.655** |
| fim_gold 0.5B | 0.150 | 0.557 | 0.645 |
| fim_mellum 0.5B | 0.156 | 0.553 | 0.643 |
| fim_mix 0.5B | **0.161** | 0.543 | 0.646 |
| Mellum-4b (teacher) | -- | **0.652** | 0.511* |

In-distribution every fine-tune helps. Out-of-distribution FIM
(HumanEval Infilling) every fine-tune slightly hurts and the 0.5B
base is the best non-teacher number. Out-of-distribution L2R
(RepoBench) the four 0.5B variants are within 1.2 ES points of each
other -- **FIM fine-tuning did not damage L2R LM ability**, which is
the regression I was checking for. Combined picture: the FIM students
learned in-distribution FIM, kept their L2R, and slightly hurt
out-of-distribution FIM. Discussion + Mellum's RepoBench caveat (*)
in the "Mellum-as-teacher seq-KD: did it work?" section of `report.md`.

## Why these metrics

Three columns because each one tells a different story. Held-out NLL is
what the training loss looks at; forward KL minimises it by construction,
so a forward-KL student wins NLL almost by definition, and the interesting
question is whether that translates downstream (it largely doesn't here).
HumanEval pass@1 is what the user actually sees: a prompt goes in, a
completion comes back, the test harness runs it, it either works or it
doesn't. Speculative-decoding draft acceptance is what determines latency
in a draft-and-verify deployment: by [Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192),
the probability of accepting a drafted token x is `min(1, p_T(x) / p_S(x))`,
and expected accepted prefix length sets the wall-clock cost. For a team
like JetBrains' Mellum, whose explicit constraint is inference cost rather
than benchmark scores, that's the metric the distillation objective should
be aimed at, not NLL or pass@1.
