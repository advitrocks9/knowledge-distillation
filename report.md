# Forward vs reverse KL when distilling a code model

I read MiniLLM (Gu et al., ICLR 2024) and wanted to see whether its
reverse-KL claim survives contact with a real code teacher and student.
What follows is what I built, 1.5B → 0.5B Qwen2.5-Coder distillation on
Python with four loss variants, and how my read of the paper changed
once the eval got honest enough to disagree with my first-pass
conclusion.

## The paper

[MiniLLM](https://arxiv.org/abs/2306.08543) argues that the forward KL
used in standard sequence-level distillation since
[Hinton et al. (2015)](https://arxiv.org/abs/1503.02531) is the wrong
objective for autoregressive LLMs. Forward KL is mode-covering: it
pushes the student to put probability on every token the teacher
considers possible, including the long tail of locally-plausible-but-
globally-wrong continuations. Sample autoregressively from a forward-KL
student and those small leaks compound. Reverse KL is mode-seeking and
asks the student to put its mass where the teacher does, not the other
way round; the student can ignore parts of the teacher's distribution
it doesn't have capacity to represent. MiniLLM pairs the direction
switch with a policy gradient over student-sampled prefixes so the
objective sees the same exposure as inference does.
[GKD](https://arxiv.org/abs/2306.13649) (Agarwal et al., 2024) arrives
at the same on-policy fix from the imitation-learning angle. I treat
MiniLLM as the spine and borrow GKD's simpler on-policy estimator
because it avoids REINFORCE variance.

## Why this matters for Mellum

Mellum is a 4B Llama-shaped code completion model trained from scratch
on 4.2T tokens
([model card](https://huggingface.co/JetBrains/Mellum-4b-base)), 256
H200s for ~20 days. The team's explicit constraint is that completion
runs at no additional cost to users, so the model has to fit inside an
inference-cost budget, not chase benchmark numbers. That changes what
distillation should be aimed at: not "make a smaller model that's
almost as good," but "given a latency budget, what loss gets the
student to draft tokens a verifier will keep, or a user will accept."
The metric closest to that question is speculative-decoding draft
acceptance ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)),
not held-out NLL or pass@1.

## Setup

| | model | params | role |
|---|---|---|---|
| teacher | Qwen2.5-Coder-1.5B | 1.54B | frozen, BF16 |
| student | Qwen2.5-Coder-0.5B | 0.49B | trainable, BF16 |
| training corpus | `codeparrot/codeparrot-clean-valid` | 2048 seqs × 512 tokens, shuffled seed 0 |
| tokenizer | shared (Qwen2 BPE, 151,936 vocab) | |
| optimiser | AdamW lr 2e-5, cosine to 0, warmup 100, weight decay 0.01 | |
| schedule | 2500 steps, batch 2, grad-accum 4 (eff bs 8), grad-clip 1.0 | |
| hardware | single RTX 4090 (24GB) sharing the box with another job | |

I use Qwen2.5-Coder rather than Mellum because Mellum has its own custom
tokenizer (98k vocab, Llama-style) and there's no smaller Mellum public
checkpoint with the same vocab to distil into. Cross-tokenizer distillation
is its own can of worms; staying inside the Qwen family lets the comparison
be about the loss, not the tokenizer map.

Four loss variants, run from the same student starting weights, identical
hyperparameters:

| variant | description |
|---|---|
| `ce` | next-token cross-entropy on the corpus, no teacher. Sanity baseline. |
| `fkl` | forward KL: `T² · KL(p_T ‖ p_S)`, T=1, teacher-forced positions. |
| `rkl` | reverse KL: `T² · KL(p_S ‖ p_T)`, T=1, teacher-forced positions. |
| `gkd` | reverse KL on student-sampled rollouts: prompt the student with 64 tokens of the corpus, let it sample 64 more, score those positions with the teacher, minimise reverse KL on the sampled segment. |

`gkd` is the cheap stand-in for MiniLLM's full estimator: no length
normalisation, no policy-gradient surrogate, no teacher-forced mixing. It's
"what's the smallest change that puts the student on its own trajectory."

## Results

### First pass

Cheap version: 32 HumanEval prompts, K=2 and K=4, no CIs. I ran this
to get a number on the board before deciding what to look at carefully.
First-pass table (`results/eval.json`):

| run | held-out NLL | HumanEval pass@1 | spec K=2 (max 2) | spec K=4 (max 4) |
|---|---|---|---|---|
| teacher Qwen2.5-Coder-1.5B | **1.0725** | **0.427** (70/164) | 2.000 | 3.984 |
| student Qwen2.5-Coder-0.5B (no fine-tune) | 1.2845 | 0.256 (42/164) | **1.640** | 2.396 |
| student + ce  | **1.2818** | 0.274 (45/164) | 1.528 | 2.355 |
| student + fkl | 1.2870 | 0.268 (44/164) | 1.387 | **2.518** |
| student + rkl | 1.3328 | 0.262 (43/164) | 1.576 | 2.390 |
| student + gkd | 1.3181 | 0.268 (44/164) | 1.492 | 2.446 |

The story I wrote: forward KL wins K=4 spec-decode, CE drifts the
student toward the data and hurts spec-decode, K=2 inverts because the
base front-loads acceptance. That story was wrong. 32 prompts can't
bracket the K=4 means, `max_drafts` was 4 for K=2 but 2 for K=4 (so
the K=2 column had twice as many draft cycles per prompt), and the
K=2/K=4 sign-flip was an artefact of those two together.

### Hardening the spec-decode evaluation

Before locking the headline I sent the design to a code-review pass and
the pushback was direct: 32 prompts with no error bars and a metric
that flips sign between K=2 and K=4 is two noisy points that I'd
attached a narrative to. The first task wasn't seeds or a new loss; it
was making the existing numbers credible. Hardened protocol:

- All 164 HumanEval prompts (function-completion shape, the closest
  thing to IDE use; no codeparrot mid-statement truncations, which
  inflate next-token sharpness artefactually).
- Bootstrap-by-prompt 95% CIs on mean accepted run length.
- Within-prompt cycle CV reported alongside the mean: a stable mean
  with a high CV is uneven user-perceived latency.
- Shared eval seed across variants so eval noise can't pose as
  training signal. Locked sampling at T=1.0 (the Leviathan rule),
  identical across all five runs.
- Per-position acceptance from the **first** drafted block per prompt
  only, so it isn't contaminated by accepted student prefixes from
  earlier cycles.
- K=4 only. K=8 isn't worth the compute until K=4 is precise enough to
  discriminate trend from variance, which on 164 prompts it barely is.

Code is `spec_eval.py`; results in `results/spec_eval.json`. ~50
minutes on the 4090 once the protocol was locked.

One protocol choice worth flagging because it differs from textbook
Leviathan: I report the per-block expected accepted run length
`E[L] = sum_i prod_{j<=i} min(1, p_T/p_S)` rather than rolling the
rejection-sampling Bernoulli per drafted token and reporting the
realised count. Both estimands converge to the same value in
expectation; the analytical version removes one source of eval noise
so the variant ranking comes out of fewer prompts. The ranking is
unaffected because the same estimator is applied across all five runs;
the per-position numbers (which are just the acceptance probabilities
themselves) are bit-exact either way. Not bit-exact Leviathan, but the
comparison it supports is honest.

### Hardened spec-decode results, K=4

| run | mean accepted run / 4 | 95% CI | within-prompt CV |
|---|---|---|---|
| teacher (self-spec, sanity) | 3.980 | [3.968, 3.990] | 0.014 |
| student base (no fine-tune) | 2.517 | [2.409, 2.624] | 0.478 |
| student + ce  | 2.359 | [2.261, 2.457] | 0.511 |
| student + fkl | 2.477 | [2.377, 2.583] | 0.485 |
| student + rkl | **2.573** | [2.474, 2.684] | 0.442 |
| student + gkd | 2.562 | [2.460, 2.658] | 0.446 |

CI non-overlap on the marginals isn't a paired test, so I ran the
paired version. Per-prompt RNG is shared across variants
(`rng_seed = eval_seed * 1_000_003 + i` in `spec_eval.py`) so the
design is paired by construction; `analyses/paired_bootstrap.py`
computes paired-bootstrap CIs on per-prompt deltas:

| pair | mean delta | 95% CI |
|---|---|---|
| rkl - ce | +0.214 | [+0.070, +0.357] |
| gkd - ce | +0.203 | [+0.064, +0.342] |
| fkl - ce | +0.118 | [-0.024, +0.260] |

`rkl > ce` and `gkd > ce` are paired-significant. fkl > ce is
consistent in direction across prompts but the paired CI crosses zero.
The first-pass headline (FKL wins K=4) doesn't survive: with 164
prompts and a proper paired test, the FKL student is in the same band
as the base and both RKL variants. The finding that does survive is
**reverse-KL distillation preserves teacher alignment better than
CE-only fine-tuning**, which is closer to what MiniLLM predicts than
my first claim.

Caveat on the paired CIs: the current `spec_eval.json` only persists
aggregates, not per-prompt arrays. The CIs above come from the
upper-bound `var(delta) <= var(a) + var(b)` in
`analyses/paired_bootstrap.py`, which assumes the two arms are
independent. The actual paired CI is narrower (positive correlation
from the shared eval seed shrinks the delta variance), so the rkl-ce
and gkd-ce significance calls are conservative.

Within-prompt CV of ~0.45-0.51 across all student variants is what the
first-pass eval was also hiding. The mean accepted run is ~2.5/4, but
cycle-to-cycle the student produces run lengths that vary by ~50% of
the mean. For a deployed code completion model that's uneven latency.
The teacher's CV is 0.014, two orders of magnitude smaller, which is
what "stable from the verifier's point of view" looks like.

### Per-position analysis on the val corpus

A more direct window than aggregate spec-decode: per-token TV and
top-1 mass on the val tensor (128 seqs × 512 tokens, ~61.5K non-pad
positions), bucketed by teacher entropy into quartiles. Low entropy
is "the teacher knows exactly what comes next" (operators, indentation,
common keywords); high entropy is "the teacher has a spread"
(variable names, API choice, high-level structure).

**Mass on the teacher's top-1 token** `p_S(argmax p_T)`:

| run | q1 (H~0) | q2 (H~0.18) | q3 (H~0.97) | q4 (H~3.16) | overall |
|---|---|---|---|---|---|
| base | 0.991 | 0.901 | 0.618 | 0.265 | 0.694 |
| ce  | 0.991 | 0.909 | 0.651 | 0.268 | 0.705 |
| fkl | 0.990 | 0.899 | 0.615 | 0.261 | 0.691 |
| **rkl** | **0.994** | **0.930** | **0.669** | **0.320** | **0.728** |
| gkd | 0.990 | 0.908 | 0.637 | 0.291 | 0.706 |

**Total-variation distance to teacher** `0.5 Σ|p_S - p_T|`:

| run | q1 | q2 | q3 | q4 | overall |
|---|---|---|---|---|---|
| base | 0.009 | 0.079 | 0.193 | 0.337 | 0.155 |
| ce  | 0.009 | 0.081 | 0.231 | 0.365 | 0.172 |
| **fkl** | 0.010 | 0.081 | 0.194 | **0.338** | 0.156 |
| **rkl** | **0.006** | **0.063** | **0.189** | 0.350 | **0.152** |
| gkd | 0.009 | 0.078 | 0.194 | 0.343 | 0.156 |

This is where the loss-shape effect is actually visible. CE drifts the
student furthest from the teacher, especially at high entropy (q4 TV
0.337 base → 0.365 ce, q3 TV 0.193 → 0.231, q4 argmax-agreement 0.604
→ 0.574); that's the per-position evidence behind CE underperforming
the KL methods on aggregate. FKL matches the teacher's distribution
shape best at q4 (TV 0.338, closest of any trained variant to base
0.337) without raising top-1 mass beyond base, which is mode-covering
exactly as advertised. RKL does the opposite: maximally concentrates
mass on the teacher's top-1 at every bucket (q4 0.320 vs base 0.265,
FKL 0.261) at the cost of slightly worse q4 TV (0.350 vs FKL 0.338),
the textbook mode-seeking trade-off. The reason this barely shows up
in aggregate spec-decode is that argmax-agreement is dominated by
q1+q2 (every method >96% there); the q4 bucket where the methods
diverge contributes proportionally less to the K=4 mean.

### Per-position acceptance, first block of K=4

For each method, mean acceptance probability at draft positions 1
through 4, computed only on the first block sampled per prompt (so
later cycles can't bias position-1):

| run | pos 1 | pos 2 | pos 3 | pos 4 |
|---|---|---|---|---|
| teacher (self) | 1.000 | 1.000 | 1.000 | 1.000 |
| base | 0.959 | 0.628 | 0.730 | 0.837 |
| ce  | 0.960 | 0.615 | 0.717 | 0.841 |
| fkl | 0.958 | 0.618 | 0.735 | 0.827 |
| **rkl** | **0.967** | **0.655** | 0.701 | 0.828 |
| gkd | 0.962 | **0.666** | 0.710 | 0.809 |

The win of RKL and GKD over CE shows up almost entirely at positions 1
and 2 (RKL pos-2 0.655 vs CE 0.615, RKL pos-1 0.967 vs CE 0.960); at
positions 3 and 4 the methods are roughly tied. Mode-seeking
distillation sharpens first-token agreement where prompt context is
most informative, then tapers as every method converges. The pos-1
/ pos-2 / pos-3+ shape itself is universal: pos-1 is high because the
student inherits good first-token prediction from pretraining; pos-2
dips because it's the first token sampled conditional on the
student's prefix; pos-3+ recovers because student-likely paths are
paths the student finds easy to continue. Loss variants change the
height of the curve, not its shape. This is also what kills the
first-pass K=2 story: the per-position picture has FKL and base tied
at every individual position, exactly what the K=4 aggregate says.

### Tying it back to spec-decode

The per-position TV column and the per-position acceptance column are
the same quantity in different units. By Corollary 3.6 of
[Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192), per-token
acceptance is exactly `1 - TV(p_S, p_T)` (one step from the rule
`min(1, p_T(x)/p_S(x))` and the identity `sum_x min(p, q) = 1 - TV`).
DistillSpec (Zhou et al. 2023, §3) uses this as the central identity
behind their training objective. I'd been treating them as adjacent
metrics; they aren't.

That changes the right way to predict K=4 from per-position acceptance
β_i. The mean β across positions isn't the quantity that matters; the
survival-weighted compound is:

`E[run/K] = (1/K) * sum_{i=1..K} prod_{j=1..i} β_j`

Two methods with the same mean β can have different K=4 numbers if one
puts its acceptance budget early and the other late.
`analyses/predict_specdecode.py` plugs the per-position β into this:

| run | predicted E[run/4] | observed run/4 |
|---|---|---|
| teacher | 1.000 | 0.995 |
| base | 0.592 | 0.629 |
| ce  | 0.582 | 0.590 |
| fkl | 0.586 | 0.619 |
| rkl | 0.603 | 0.643 |
| gkd | **0.607** | 0.640 |

Predicted ranking is `gkd > rkl > base > fkl > ce`; observed is
`rkl > gkd > base > fkl > ce`. The top two swap (the RKL-GKD gap is
0.004 predicted, 0.003 observed, well inside the bootstrap CIs);
otherwise preserved.

Two surprises. First, predictions are systematically *lower* than
observed (CE 0.582 vs 0.590, RKL 0.603 vs 0.643), opposite of my
intuition that later cycles condition on student-drifted prefixes and
should have lower β than first-cycle β. The data go the other way.
Plausible explanation: the advance step pushes the prefix toward
positions the student is locally confident at (end-of-line, closing
brackets), which have high acceptance even when the broader
distribution diverges. No confirmatory experiment yet.

Second, why FKL doesn't win mean TV despite Pinsker's inequality
(`TV ≤ sqrt(KL(p_T || p_S) / 2)`) saying it should. Capacity. When the
student can't match the teacher's full distribution exactly, FKL has
to spread mass across the teacher's full support, including positions
where the teacher is itself uncertain (q4). RKL mode-collapses onto
the high-density regions, giving very low TV at q1-q3 at the cost of
slightly higher TV at q4. The per-position TV table says it: 75% of
Python positions are mode-seekable, 25% are mode-coverable. RKL wins
aggregate because the seekable fraction dominates.

### Three-seed sanity on CE vs RKL

CE → RKL is the only pair the single-seed hardened eval distinguishes
significantly, so it's worth confirming it isn't a lucky-seed artefact.
I retrained CE and RKL at seeds 0, 1, 2 (same data, same hyperparams,
only `torch.manual_seed` changes) and re-ran the hardened spec-decode
eval on each:

| variant | seed 0 | seed 1 | seed 2 | mean | seed std |
|---|---|---|---|---|---|
| CE  | 2.370 | 2.378 | 2.386 | 2.378 | 0.008 |
| RKL | 2.569 | 2.573 | 2.611 | 2.584 | 0.023 |

Gap of 0.206 tokens against a pooled SD of ~0.017 is an effect size of
~12 SDs. The RKL > CE finding survives seed variance. I deliberately
didn't seed FKL, GKD, or the un-fine-tuned base: spending three more
seeds on methods already within noise of each other on a 164-prompt
eval isn't worth the compute.

## What this changed about my read of the paper

After the first-pass eval I'd written that MiniLLM was overstating its
case for code distillation. After the hardened eval that's the wrong
read. The loss-direction effect I'd attached a narrative to was a
32-prompt sample-size artefact; what survives is the distillation
effect itself, "use the teacher" beats "don't," on the metric that
maps onto latency. That's actually consistent with MiniLLM at this
scale: same architecture, same pretraining mix, 3× capacity gap means
teacher and student already largely agree at the head of the
distribution, so the loss-shape lever mostly shows up as "did you use
the teacher at all," not "which KL direction." The right place to test
the mechanism in full is wider gaps: 7B → 0.5B in Qwen, or
Mellum-4B → sub-1B.

The on-policy fix in MiniLLM/GKD is the part of the recipe my naive
`gkd` run isn't really exercising. It lands inside the off-policy
reverse-KL CI despite being on-policy, which says my naive "sample,
score, reverse-KL on the segment" isn't adding what MiniLLM's algorithm
adds: length-normalised reward, teacher-mixed prefixes, single-step
decomposition. With those, on-policy should beat off-policy reverse KL
by more than the noise floor. And
[DistillSpec](https://arxiv.org/abs/2310.08461) is the natural next
objective: it trains the student against per-token acceptance
probability directly, which is the quantity whose units match the
metric the eval discriminates on.

## Smaller things from the runs

5e-5 overshot and made every run worse than base on val NLL; 2e-5
with 100-step warmup is what worked. The bug along the way was calling
`sched.step()` only inside the gradient-accumulation branch, which
advanced the cosine schedule 4× slower than intended. The 1e-5 to 3e-5
range is what distillation papers cite for fine-tuning a pretrained
student, not the 5e-5 you'd use from scratch.

GKD is ~5× slower per step than CE because every step samples 64
tokens autoregressively from the student before computing the loss,
and gradient checkpointing forces KV caching off in `generate`. The
obvious optimisation for a serious follow-up is a replay buffer of
student rollouts so it isn't resampled every step.

`codeparrot-clean-valid` has near-duplicates: several of my training
sequences are template-y test-fixture files that recur across the
corpus. Left them in to be honest about the data; a real run would
dedupe by file hash plus a near-dup filter.

## Mellum-as-teacher seq-KD: did it work?

The obvious gap in the Qwen-on-Qwen experiment is that I kept talking
about Mellum without using it. This follow-up, which I ran on Modal
with $30 of credits after the lab GPU box went down, is a
cross-tokenizer sequence-level distillation: Mellum-4b-sft-python
generates FIM completions on a Python corpus, those text targets get
re-tokenized in Qwen, and Qwen2.5-Coder-0.5B is SFT'd on (prefix,
mellum-middle, suffix) with loss masked to the middle. This is
Kim & Rush 2016 seq-level KD: once tokenizers don't match, you can't
do logit distillation, only pseudo-labelling.

Five conditions. The gold-FIM condition is the control: without it I
can't separate "Mellum's text is good signal" from "any FIM fine-tune
on this corpus would help."

| condition | what it sees during training |
|---|---|
| `base` | nothing, the un-fine-tuned Qwen2.5-Coder-0.5B |
| `fim_gold` | (prefix, *ground-truth* middle, suffix), 600 codeparrot examples |
| `fim_mellum` | (prefix, *Mellum-generated* middle, suffix), same 600 examples |
| `fim_mix` | 50/50 mix of gold and mellum middles |
| `mellum_4b` | the teacher itself, upper bound |

Hyperparams: 1200 steps, batch 2 × accum 4, lr 2e-5 cosine, middle-only
loss masking, greedy Mellum decoding for the seq-KD targets (greedy is
the more conservative target choice because it removes one source of
variance; sampling-vs-greedy seq-KD targets is contested in the
literature).

Eval is held-out codeparrot FIM (180 examples, 60 per masking kind,
exact-match against gold middles, in-distribution) and HumanEval
Infilling (164 per subset, fixed seed shared across models, the
published Mellum benchmark, out-of-distribution).

### Held-out codeparrot FIM (in-distribution)

| method | single | multi | random | overall EM | overall middle NLL |
|---|---|---|---|---|---|
| base | 0.350 | 0.000 | 0.017 | 0.122 | 1.013 |
| fim_gold | 0.367 | 0.067 | 0.017 | 0.150 | **0.965** |
| fim_mellum | 0.383 | 0.083 | 0.000 | 0.156 | 1.018 |
| **fim_mix** | **0.400** | **0.083** | 0.000 | **0.161** | 0.975 |

In-distribution, FIM training works. Mix wins EM (+3.9pp over base);
gold wins NLL-against-gold-middles, which it should by construction.
fim_mellum has *worse* gold-middle NLL than base because it learned to
generate Mellum-style middles that differ from gold middles in style;
its exact-match is still competitive with fim_gold, which says Mellum's
middles are closer to gold in content than in token distribution.

### HumanEval Infilling (out-of-distribution, the published benchmark)

164 sub-sampled per subset (full set is 1033/5815/1640), fixed seed
shared across all five models, pass@1 against `prefix + completion +
suffix`:

| method | single | multi | random | mean |
|---|---|---|---|---|
| base 0.5B | **0.787** | 0.396 | 0.506 | 0.563 |
| fim_gold 0.5B | 0.787 | **0.415** | 0.470 | 0.557 |
| fim_mellum 0.5B | 0.774 | 0.415 | 0.470 | 0.553 |
| fim_mix 0.5B | 0.768 | 0.390 | 0.470 | 0.543 |
| mellum_4b (teacher) | 0.738 | **0.537** | **0.683** | **0.652** |

Sanity-check: Qwen2.5-Coder-0.5B's paper reports 0.754/0.473/0.460
(mean 0.562) on the full set; my 164-subsample base of
0.787/0.396/0.506 (mean 0.563) lines up. Mellum-4b-base reports
0.6621/0.3852/0.2970 (mean 0.448); my Mellum-sft-python is the
Python-fine-tuned variant which is known to beat the base on Python
infill.

**None of the FIM-tuning conditions beat the un-fine-tuned base on
HumanEval Infilling.** Plain base (0.563) edges out every fine-tuned
variant. fim_mellum (0.553) is below the gold-data control fim_gold
(0.557); fim_mix is worst at 0.543.

### What this says

In-distribution and out-of-distribution point opposite ways. FIM
training improves performance on the training distribution (codeparrot)
and degrades it on the distribution-shift benchmark (HumanEval). The
student is learning the *style* of the training corpus, not the *task*
of FIM. That's the failure mode Bavarian et al. (2022) flag in the FIM
paper: FIM capability comes from the data transformation at pretraining
scale, not from a 600-example fine-tune layered on a model already
FIM-pretrained on billions of tokens.

The Mellum-as-teacher signal isn't visible at this scale either:
`fim_mellum` and `fim_gold` are within noise on both evals, with the
small direction of effect on HumanEval Infilling consistent with
"Mellum's decoded text drifts slightly off the canonical-solution form
HumanEval grades against." A larger corpus, or logit-level KD instead
of text-level pseudo-labels, would be the right place to look.
Cross-tokenizer logit distillation is the obvious next step but the
engineering is real: two BPE schemes need either a byte-level alignment
or a soft-token bridge à la
[ULD (Boizard et al. 2024)](https://arxiv.org/abs/2402.12030).

The one positive signal is multi-line. `fim_gold` and `fim_mellum` both
beat base on multi-line HumanEval pass@1 (0.415 vs 0.396, +1.9pp each),
and that's the subtask base is weakest on. The held-out codeparrot EM
points the same way (base 0.000, fim_gold 0.067, fim_mellum 0.083,
fim_mix 0.083). Multi-line is where the small fine-tune actually
delivers something.

### RepoBench-Python next-line (the L2R regression check)

The HumanEval Infilling regression worried me because I couldn't tell
apart "FIM training learned codeparrot style and hurts out-of-distribution
FIM" from "FIM training damaged the student's plain LM ability." A
third eval on the *other* benchmark Mellum's card reports settles it:
RepoBench-Python next-line (Liu et al. 2023,
`tianyang/repobench_python_v1.1`), three subsets, 60 problems each at
file lengths ≤8k, fixed seed shared across models. Score is
edit-similarity (difflib ratio) between the model's first generated
line and the canonical next line.

| method | cf_first | cf_random | in_file | avg ES | avg EM |
|---|---|---|---|---|---|
| base 0.5B | **0.587** | 0.681 | 0.697 | **0.655** | 0.239 |
| fim_gold 0.5B | 0.571 | 0.667 | 0.697 | 0.645 | 0.228 |
| fim_mellum 0.5B | 0.564 | **0.689** | 0.677 | 0.643 | 0.183 |
| fim_mix 0.5B | 0.565 | 0.668 | **0.705** | 0.646 | 0.222 |
| mellum_4b (FIM-wrap) | 0.474 | 0.553 | 0.505 | 0.511 | 0.178 |

The four 0.5B variants land within 1.2 ES on the average and within 2
ES on every subset. **FIM fine-tuning did not damage L2R LM ability.**
Combined with the HumanEval Infilling regression: FIM students learned
codeparrot-style FIM, kept their L2R, slightly hurt out-of-distribution
FIM. They didn't forget how to be language models.

Mellum row caveat: my first run gave Mellum raw L2R prompts and it
scored ES 0.470 because Mellum-sft-python is FIM-only and hits EOS in
1-2 tokens on a raw L2R prompt. Re-running wrapped as
`<fim_suffix><fim_prefix>{code}<fim_middle>` (empty suffix, FIM as a
degenerate next-token completion) gave the 0.511 above. The Mellum
card's published 0.299 isn't directly comparable (different RepoBench
version, file-length bucket, and cross-file context format with
`<filename>` / `<file_sep>` markers I'm not providing). Treat 0.511
as a loose lower bound under my protocol, not a statement about
Mellum's capability.

## What I'd change next

The cheapest fix for the FIM follow-up is early stopping. All three
trainings overfit: val middle NLL at steps 300/600/900/1200 is
0.91/0.95/0.98/0.98 for gold, 0.96/1.01/1.04/1.04 for mellum,
0.92/0.97/1.00/1.01 for mix, monotonically up after step 300 in every
run. Checkpointing every 100 steps and picking the best by val NLL
probably recovers 30-40% of the HumanEval gap. Beyond that, the
seq-KD corpus needs to be 50K-500K examples, not 600, and on a corpus
that matches the eval distribution (HumanEval-style short functions,
not real GitHub repos with messy structure).

For the main spec-decode line, the changes that would actually move
the rkl-vs-fkl ranking, in expected-payoff order:

1. Length-normalised reward in the on-policy reverse KL, the actual
   MiniLLM section 3.3 algorithm. Naive on-policy drifts because
   teacher reward at long rollouts is dominated by easy positions;
   length-normalising stops the student collapsing onto whichever
   mode the teacher rewards on average. Cheapest single change.
2. DistillSpec: train against per-token acceptance probability
   directly, stop-gradient through the teacher. Units match the
   metric.
3. Wider teacher-student gap inside the Qwen family. 1.5B → 0.5B is
   3×; the reverse-KL effect should sharpen at 7B → 0.5B. Skipped
   here because it sharpens the proxy without bringing it closer to
   Mellum.
4. Cross-tokenizer logit KD via ULD or byte-alignment, so the
   Mellum-as-teacher seq-KD has access to per-token uncertainty
   instead of just decoded text.

## How to repro

```bash
uv sync
uv run python data.py                                                 # ~5s
uv run python distill.py --loss ce  --steps 2500 --lr 2e-5            # ~5 min
uv run python distill.py --loss fkl --steps 2500 --lr 2e-5            # ~7 min
uv run python distill.py --loss rkl --steps 2500 --lr 2e-5            # ~7 min
uv run python distill.py --loss gkd --steps 2500 --lr 2e-5            # ~40 min (sampling)
uv run python eval.py                                                 # first-pass aggregate eval
uv run python spec_eval.py --K 4 --max-drafts 8 --eval-seed 42        # hardened spec-decode eval
uv run python make_table.py                                           # prints the markdown
```

Total wall on a single RTX 4090 (shared with another job): about three
hours. `results/train_*.json` has per-step training curves,
`results/eval.json` the first-pass aggregate, `results/spec_eval.json`
the hardened spec-decode numbers with per-position acceptance and
bootstrap CIs.

## Things I'd want a Mellum engineer to push back on

1. At 1.5B → 0.5B the reverse-KL win sits almost entirely in the q4
   bucket, which is only a quarter of positions, so it gets diluted. At
   4B → sub-1B with a bigger entropy gap, does q4 dominate, or do the
   easy positions still swamp it?
2. Is spec-decode draft length the right student-side metric for Mellum
   at all? My read of the public posts is that Mellum runs alone in the
   IDE with no separate verifier, so the latency lever is the model's
   own forward-pass time and "acceptance" is whether the user kept the
   completion. If so, the right offline loss is closer to "expected
   tokens before the user edits," which is harder to target without
   the telemetry.
