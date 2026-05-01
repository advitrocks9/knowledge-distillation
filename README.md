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

The two pairs whose CIs don't overlap, i.e. statistically distinguishable
at 95%: `rkl > ce` and `gkd > ce`. Everything else is within noise.

Three-seed retraining of CE and RKL (the pair that survived the
single-seed eval) confirms the gap: CE mean 2.378 (seed std 0.008),
RKL mean 2.584 (seed std 0.023), effect size ~12 pooled SDs. See
`results/spec_eval_seeds.json`.

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
python distill.py --loss ce  --steps 2500 --lr 2e-5  # baseline
python distill.py --loss fkl --steps 2500 --lr 2e-5  # forward KL
python distill.py --loss rkl --steps 2500 --lr 2e-5  # reverse KL
python distill.py --loss gkd --steps 2500 --lr 2e-5  # on-policy reverse KL
python eval.py                                       # HumanEval pass@1, NLL, spec-decode
python make_table.py                                 # markdown table
```

About 2h on a single 4090. The `gkd` step does autoregressive student
sampling, which is most of the training time; everything else fits in 8 min.

## What's in here

```
data.py        codeparrot Python -> torch tensor cache
distill.py     one script, four losses (ce, fkl, rkl, gkd)
eval.py        HumanEval pass@1, held-out NLL, spec-decode draft acceptance
make_table.py  pretty-prints results/eval.json
results/       per-run training logs and eval.json
report.md      writeup with the discussion
```

## Why these metrics

Held-out NLL is the loss-shaped metric. Forward KL minimises it by
construction so a forward-KL student wins NLL almost by definition; the
question is whether that translates downstream. It largely doesn't, in this
experiment.

HumanEval pass@1 is the user-facing metric. The student gets a Python prompt,
emits a completion, the test harness runs it. Either it works or it doesn't.

Speculative-decoding draft acceptance is the latency-shaped metric. Per
[Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192), the probability
of accepting a drafted token x is `min(1, p_T(x) / p_S(x))`, and the expected
length of the accepted prefix is what determines wall-clock for a draft-and-
verify setup. For a team like JetBrains' Mellum, whose explicit constraint is
inference cost rather than benchmark scores, this is the metric the
distillation objective should be aimed at -- not NLL, not pass@1.
