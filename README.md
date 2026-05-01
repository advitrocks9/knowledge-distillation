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

## Headline numbers (full table in `report.md`)

| run | NLL | HumanEval pass@1 | spec-decode K=4 |
|---|---|---|---|
| teacher 1.5B | 1.0725 | 0.427 | 3.984 / 4 |
| student base | 1.2845 | 0.256 | 2.396 / 4 |
| ce baseline | **1.2818** | **0.274** | 2.355 / 4 |
| forward KL | 1.2870 | 0.268 | **2.518 / 4** |
| reverse KL | 1.3328 | 0.262 | 2.390 / 4 |
| GKD on-policy | 1.3181 | 0.268 | 2.446 / 4 |

Each loss wins a different column. CE wins NLL and HumanEval by a small
margin; forward KL wins the speculative-decoding draft acceptance metric (the
one that maps onto inference latency for a code completion model); reverse KL
wins nothing at this teacher size and training duration. Discussion in
`report.md`.

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
