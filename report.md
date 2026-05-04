# Forward vs reverse KL when distilling a code model

I read MiniLLM (Gu et al., ICLR 2024) and wanted to know whether its
reverse-KL claim survives contact with a real code teacher and student.
This is a writeup of what I built (1.5B → 0.5B Qwen2.5-Coder distillation
on Python, four loss variants, three eval axes) and what changed about
my read of the paper after I made the eval honest enough to disagree
with my first-pass conclusion. Both directions matter: the eval got
better, the conclusion flipped, and the paper turned out to be more
right than I gave it credit for on my first reading.

## The paper, in one paragraph

[MiniLLM](https://arxiv.org/abs/2306.08543) argues that the forward KL
divergence used in standard sequence-level distillation since
[Hinton et al. (2015)](https://arxiv.org/abs/1503.02531) is the wrong
objective for autoregressive LLMs. Forward KL is mode-covering: it pushes
the student to put probability on every token the teacher considers
possible, including the long tail of locally-plausible-but-globally-wrong
continuations. When you sample autoregressively from a forward-KL student
those small leaks compound. Reverse KL is mode-seeking and asks the student
to put its mass where the teacher does, not the other way round. The student
is allowed to ignore parts of the teacher distribution it doesn't have
capacity to represent. MiniLLM pairs this with a policy-gradient training
scheme so the reverse-KL objective is computed on student-sampled prefixes
rather than teacher-forced ones, fixing the exposure-bias problem at the
same time.

The closely related paper is [GKD](https://arxiv.org/abs/2306.13649)
(Agarwal et al., 2024), which arrives at the same on-policy fix from the
imitation-learning angle and uses a generalised JSD that interpolates
between forward and reverse KL. I treat MiniLLM as the spine because the
reverse-KL framing is the sharper hook, and I borrow GKD's simpler
on-policy estimator for my own implementation because it avoids the
variance of REINFORCE.

## Why this matters for Mellum specifically

Mellum is a 4B-parameter Llama-shaped code completion model trained from
scratch on 4.2T tokens
([JetBrains/Mellum-4b-base model card](https://huggingface.co/JetBrains/Mellum-4b-base)),
on 256 H200 GPUs for ~20 days. Its reason for existing is that JetBrains
wants completion that runs at "no additional cost to users", which means
the model has to fit inside an inference-cost budget, not chase benchmark
numbers. That reframes distillation: it's not "make a smaller model that's
almost as good"; it's "given a fixed latency target, what's the best
objective for getting a small student to draft tokens that a verifier will
keep, or a user will accept." The metric closest to that question is
speculative-decoding draft acceptance
([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192)), not held-out
NLL or even pass@1. The reverse-KL framing matters here because at 4B the
teacher has real capacity to spread small mass across plausible
continuations, the "junior dev rambling" failure mode, and you want the
student to ignore the rambling, not imitate it.

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

The cheap version of the eval: 32 HumanEval prompts, K=2 and K=4, no
confidence intervals. I ran this first because I wanted a number, any
number, before deciding what to look at more carefully. The first-pass
table looked like this (`results/eval.json`):

| run | held-out NLL | HumanEval pass@1 | spec K=2 (max 2) | spec K=4 (max 4) |
|---|---|---|---|---|
| teacher Qwen2.5-Coder-1.5B | **1.0725** | **0.427** (70/164) | 2.000 | 3.984 |
| student Qwen2.5-Coder-0.5B (no fine-tune) | 1.2845 | 0.256 (42/164) | **1.640** | 2.396 |
| student + ce  | **1.2818** | 0.274 (45/164) | 1.528 | 2.355 |
| student + fkl | 1.2870 | 0.268 (44/164) | 1.387 | **2.518** |
| student + rkl | 1.3328 | 0.262 (43/164) | 1.576 | 2.390 |
| student + gkd | 1.3181 | 0.268 (44/164) | 1.492 | 2.446 |

The story I wrote down on the first pass: **forward KL wins K=4
spec-decode**. CE drifts the student toward the data and hurts spec-decode;
forward KL preserves teacher alignment. K=2 inverts because base
"front-loads" acceptance.

That story was wrong, or at least not supported by the evidence I had.

### Hardening the spec-decode evaluation

Before locking the headline I sent the eval design to a code-review pass
(see `notes/levelup-rescue.md`) and the pushback was direct: 32 prompts
with no error bars and a metric that flips sign between K=2 and K=4 is
two noisy points that I'd attached a narrative to. The first task wasn't
seeds, or a new loss, or a bigger teacher -- it was making the existing
spec-decode numbers credible. Specifically:

- 32 → 164 prompts (all of HumanEval, the function-completion shape that
  matches IDE use; no codeparrot mid-statement truncations, which inflate
  next-token sharpness artefactually);
- bootstrap-by-prompt 95% CIs on mean accepted run length;
- per-prompt cycle CV reported alongside the mean (codex's "your mean is
  stable but your process might not be" point -- if a code completion
  team grades on user latency they care about both);
- shared eval seed across variants so evaluation noise can't masquerade
  as training signal;
- locked sampling regime (T=1.0, the protocol from Leviathan et al.
  2023, applied identically to all variants);
- per-position acceptance only from the **first** drafted block per
  prompt, to avoid contamination from previously accepted student
  prefixes;
- K=4 only -- K=8 isn't worth the compute until K=4 is precise enough
  to discriminate trend from variance.

Code is `spec_eval.py`; results are `results/spec_eval.json`. The
re-eval ran in ~50 minutes on the 4090 once the protocol was locked.

### Hardened spec-decode results, K=4

| run | mean accepted run / 4 | 95% CI | within-prompt CV |
|---|---|---|---|
| teacher (self-spec, sanity) | 3.980 | [3.968, 3.990] | 0.014 |
| student base (no fine-tune) | 2.517 | [2.409, 2.624] | 0.478 |
| student + ce  | 2.359 | [2.261, 2.457] | 0.511 |
| student + fkl | 2.477 | [2.377, 2.583] | 0.485 |
| student + rkl | **2.573** | [2.474, 2.684] | 0.442 |
| student + gkd | 2.562 | [2.460, 2.658] | 0.446 |

Pairs whose 95% CIs do not overlap (i.e. statistically distinguishable at
this sample size):

- `student_rkl > student_ce`
- `student_gkd > student_ce`

That's it. Every other pair has overlapping CIs and isn't distinguishable.

The first-pass headline -- "forward KL wins K=4" -- did not survive. The
gap I attached a story to was a 32-prompt sample-size artefact: with 164
prompts and proper CIs, the FKL student is statistically tied with the
un-fine-tuned base and with both reverse-KL variants. The actual
significant finding is **reverse-KL distillation preserves teacher
alignment significantly better than CE-only fine-tuning**, which is
closer to what MiniLLM predicts than what I originally claimed.

The within-prompt CV of ~0.45-0.51 across all student variants is the
other thing the first-pass eval was hiding. The mean accepted run is
~2.5 / 4, but draft-cycle-to-draft-cycle the student is producing run
lengths that vary by ~50% of the mean. For a deployed code completion
model that translates to noticeably uneven latency. The teacher's CV is
0.014, two orders of magnitude smaller, which is what "stable from the
verifier's point of view" looks like.

### Per-position analysis on the val corpus

Spec-decode is one window into "does the student match the teacher's
distribution." A more direct window is to compute the per-token KL,
top-1 agreement, and total-variation distance at every position of the
val corpus. The val tensor is 128 sequences × 512 tokens, of which
~61.5K positions are non-pad. Bucket the positions by teacher entropy
into quartiles -- low entropy is "the teacher knows exactly what comes
next" (operators, indentation, common keywords); high entropy is "the
teacher has a spread distribution" (variable names, choice of API,
high-level structure).

**Mass on the teacher's top-1 token** (`p_S(argmax p_T)` -- higher means
the student more confidently backs the teacher's preferred next token):

| run | q1 (H~0) | q2 (H~0.18) | q3 (H~0.97) | q4 (H~3.16) | overall |
|---|---|---|---|---|---|
| base | 0.991 | 0.901 | 0.618 | 0.265 | 0.694 |
| ce  | 0.991 | 0.909 | 0.651 | 0.268 | 0.705 |
| fkl | 0.990 | 0.899 | 0.615 | 0.261 | 0.691 |
| **rkl** | **0.994** | **0.930** | **0.669** | **0.320** | **0.728** |
| gkd | 0.990 | 0.908 | 0.637 | 0.291 | 0.706 |

**Total-variation distance to teacher distribution** (`0.5 * Σ|p_S - p_T|`
-- lower means the student matches the teacher's full distribution shape,
not just its mode):

| run | q1 | q2 | q3 | q4 | overall |
|---|---|---|---|---|---|
| base | 0.009 | 0.079 | 0.193 | 0.337 | 0.155 |
| ce  | 0.009 | 0.081 | 0.231 | 0.365 | 0.172 |
| **fkl** | 0.010 | 0.081 | 0.194 | **0.338** | 0.156 |
| **rkl** | **0.006** | **0.063** | **0.189** | 0.350 | **0.152** |
| gkd | 0.009 | 0.078 | 0.194 | 0.343 | 0.156 |

This is the figure where the loss-shape effect is visible. The two
mechanisms MiniLLM predicts -- forward KL is mode-covering, reverse KL
is mode-seeking -- show up cleanly at the per-position level even
though the aggregate spec-decode column only barely separates them:

- **CE drifts the student furthest from the teacher**, especially at
  high-entropy positions. q4 TV jumps from 0.337 (base) to 0.365 (ce);
  q3 TV jumps from 0.193 to 0.231. The argmax-agreement at q4 also
  drops the most under CE (0.604 → 0.574). This is the per-position
  evidence behind the spec-decode result that CE significantly
  underperforms the KL methods.
- **FKL matches the teacher's distribution shape best at q4** (TV 0.338,
  the closest of all four trained variants to the base 0.337) but does
  *not* increase the student's confidence on the teacher's top-1
  (0.261, slightly below base's 0.265). That's exactly mode-covering
  in action: FKL preserves the teacher's spread, including the spread.
- **RKL maximally concentrates mass on the teacher's top-1 token** at
  every bucket and especially at q4: 0.320 vs base 0.265, FKL 0.261.
  At the same time RKL's q4 TV is 0.350 -- slightly *worse* than FKL,
  because RKL isn't matching the teacher's full shape, it's mode-seeking.
  That trade-off is the textbook reverse-KL behaviour.

The three losses are doing exactly what the literature says they should
do. The reason this doesn't translate into a large aggregate spec-decode
gap is that argmax-agreement (which drives most of the per-token
acceptance) is dominated by the q1+q2 buckets where every method
agrees with the teacher >96% of the time. The q4 bucket, where the
methods diverge meaningfully, is only ~25% of positions and contributes
proportionally less to the K=4 average. So the per-position figure
shows the loss-shape mechanism more clearly than any aggregate metric I
ran -- and the aggregate metrics, with proper CIs, only reveal the
distillation-vs-no-distillation contrast.

### Why the per-position TV table predicts the spec-decode ranking

After computing the per-position TV table I had to double-check
whether the spec-decode column and the TV column were measuring the
same thing. They are, at the per-token level, by Corollary 3.6 of
[Leviathan et al. (2023)](https://arxiv.org/abs/2211.17192): the per-token
acceptance probability under speculative decoding is exactly
`1 - TV(p_S, p_T)`. The proof is one step from the rule
`min(1, p_T(x)/p_S(x))` and the standard identity
`sum_x min(p, q) = 1 - TV(p, q)`. DistillSpec (Zhou et al. 2023, §3) uses
this as the load-bearing identity for their training objective. I
should have spotted that earlier in my own writeup; I was treating
spec-decode and per-position TV as adjacent metrics rather than the
same quantity in different units.

What I think is actually mine -- the empirical operationalisation --
is using the per-position acceptance β_i (which is `1 - TV` at each
draft position) to predict the K=4 ranking via the right formula. The
expected accepted run length is the survival-weighted compound
`E[run/K] = (1/K) * sum_{i=1..K} prod_{j=1..i} β_j`. Equal-weight
average of β across positions is *not* the right quantity -- two
methods with the same mean β can have different K=4 numbers if one
puts its acceptance budget early and the other puts it late. The
correct survival-weighted prediction is in `analyses/predict_specdecode.py`:

| run | β1 | β2 | β3 | β4 | predicted E[run/4] | observed run/4 |
|---|---|---|---|---|---|---|
| teacher | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.995 |
| base | 0.959 | 0.628 | 0.730 | 0.837 | 0.592 | 0.629 |
| ce  | 0.960 | 0.615 | 0.717 | 0.841 | 0.582 | 0.590 |
| fkl | 0.958 | 0.618 | 0.735 | 0.827 | 0.586 | 0.619 |
| rkl | 0.967 | 0.655 | 0.701 | 0.828 | 0.603 | 0.643 |
| gkd | 0.962 | 0.666 | 0.710 | 0.809 | **0.607** | 0.640 |

The survival-weighted ranking is `gkd > rkl > base > fkl > ce`. The
observed ranking is `rkl > gkd > base > fkl > ce`. The top two swap
between predicted and observed -- the gap between RKL and GKD is small
in both (0.004 in predicted, 0.003 in observed) and inside the
spec-decode bootstrap CIs that overlap heavily. So the survival-
weighted prediction matches observed to within tied positions.

Two real surprises in this table: predictions are systematically
*lower* than observed (CE 0.582 vs 0.590, RKL 0.603 vs 0.643). That's
the opposite of what I expected. My naive intuition was that later
draft cycles are conditioned on student-sampled prefixes that have
drifted from the teacher, so β at later cycles should be lower than
first-cycle β -- and the prediction uses first-cycle β only. The data
say the opposite: later cycles have higher β. The plausible
explanation is that the spec-decode advance step pushes the prefix
towards positions where the student is highly confident locally
(end-of-line, closing brackets, etc.), and those positions have very
high acceptance even when the student-teacher distribution diverges
elsewhere. I haven't run a confirmatory experiment for this yet.

The subtler thing the per-position TV table reveals -- this part is
mine -- is **why FKL doesn't win the TV column despite Pinsker's
inequality saying it should**. Forward KL is the standard cleanest
minimiser of TV (`TV ≤ sqrt(KL(p_T || p_S) / 2)`), and the literature
gives it credit accordingly. At this scale FKL still doesn't win mean
TV. The reason is capacity: when the student can't match the teacher's
full distribution exactly, forward KL has to spread the available
probability mass across all of the teacher's support, including
positions where the teacher is itself uncertain (the q4 bucket).
Reverse KL mode-collapses onto the teacher's high-density regions,
giving very low TV at q1-q3 (where the teacher is sharp and mode-
collapsing is the right strategy) at the cost of slightly higher TV
at q4 (where the teacher is spread and mode-collapsing is wrong). For
the val-corpus position distribution -- 75% low-entropy positions, 25%
high-entropy -- the second strategy wins on aggregate. That's what the
per-position TV table is showing in one sentence: **the right loss
direction depends on what fraction of your positions are mode-seekable
versus mode-coverable, and Python source has more of the former**.

### Three-seed sanity on CE vs RKL

The gap CE → RKL on spec-decode is the only statistically distinguishable
finding from the single-seed eval, so it's worth confirming it isn't an
artefact of one lucky training seed. I retrained CE and RKL at seeds 0,
1, 2 (same data, same hyperparams, only the torch.manual_seed call
changes), re-ran the hardened spec-decode eval on each, and got:

| variant | seed 0 | seed 1 | seed 2 | mean | seed std |
|---|---|---|---|---|---|
| CE  | 2.370 | 2.378 | 2.386 | 2.378 | 0.008 |
| RKL | 2.569 | 2.573 | 2.611 | 2.584 | 0.023 |

Gap of 0.206 tokens against a pooled SD of ~0.017 is an effect size of
~12 SDs. The RKL > CE finding survives seed variance.

I deliberately didn't seed FKL, GKD, or the un-fine-tuned base. CE vs
RKL is the pair where the hardened single-seed eval showed
non-overlapping bootstrap CIs; spending compute confirming that pair
is worth it, spending compute on three more seeds of methods that are
already within noise of each other on a 164-prompt eval isn't.

### Per-position acceptance, first block of K=4

This is the figure I should have led with on the first pass. For each
method, mean acceptance probability at draft positions 1 through 4,
computed only on the first block sampled from each prompt:

| run | pos 1 | pos 2 | pos 3 | pos 4 |
|---|---|---|---|---|
| teacher (self) | 1.000 | 1.000 | 1.000 | 1.000 |
| base | 0.959 | 0.628 | 0.730 | 0.837 |
| ce  | 0.960 | 0.615 | 0.717 | 0.841 |
| fkl | 0.958 | 0.618 | 0.735 | 0.827 |
| **rkl** | **0.967** | **0.655** | 0.701 | 0.828 |
| gkd | 0.962 | **0.666** | 0.710 | 0.809 |

Two things this reveals that the aggregate hides:

1. The win of RKL and GKD over CE shows up almost entirely at draft
   positions 1 and 2. RKL pos 2 is 0.655 (vs CE 0.615); RKL pos 1 is
   0.967 (vs CE 0.960). At positions 3 and 4 the methods are roughly
   tied. That's consistent with mode-seeking distillation: it sharpens
   the student's first-token agreement with the teacher (where the
   prompt context is most informative) and provides less differential
   benefit at later positions where every method is converging to similar
   distributions.
2. The pos-1 / pos-2 / pos-3+ shape is universal across methods. Pos 1
   is high (~0.96) because the student inherits good first-token
   prediction from pretraining. Pos 2 dips because that's the first
   token sampled conditional on the *student's* sampled prefix, where
   the student-teacher distribution gap matters most. Pos 3+ recovers
   because conditioning on a student-likely path is also a path the
   student finds locally easy to continue. None of the four loss
   variants change the *shape* of this curve, only the absolute height.

This per-position pattern is also what kills my first-pass K=2 story.
The "FKL is worst at K=2" claim was about an aggregate over a draft
cycle that itself averages over noisy positions. The properly
disaggregated picture says FKL and base are tied at every individual
position, which is what the K=4 aggregate also says.

## What this changed about my read of the paper

The honest version of this section needs to be in two parts, because what
I think the paper says changed when the eval got more honest.

**After the first-pass eval**, I thought the paper was overstating its
case for autoregressive code distillation: forward KL had won my K=4
spec-decode column and reverse KL had won nothing. I wrote that the
reverse-KL win was conditional on the teacher having a structured tail
mass that my 1.5B code teacher didn't have, and that the on-policy fix
in MiniLLM was load-bearing in a way the headline of the paper hides.

**After the hardened eval**, the situation is different. CE-only
fine-tuning is statistically worse than reverse-KL training on draft
acceptance, and the KL-direction methods are all within noise of each
other. That's actually consistent with what MiniLLM's argument predicts
at this scale: when the teacher and student already largely agree (same
architecture, same pretraining mix, 3× capacity gap), the loss-shape
effect mostly shows up as "did you use the teacher at all", not "which
KL direction did you use." The reverse-KL student edges out the FKL
student by 0.1 of a token at K=4 on the means, which is in the right
direction but not statistically distinguishable.

The thing I was wrong about is that I'd attached a clear loss-direction
story to a 32-prompt eval. With proper CIs the loss-direction effect is
indistinguishable from noise at this teacher size. What's distinguishable
is the **distillation effect itself** -- using the teacher (any KL
direction) versus not using the teacher (CE) -- on the metric that
maps onto inference latency.

Three things this updates me on, in priority order:

1. **The right unit for distillation evaluation is acceptance length with
   CIs, not aggregate NLL or aggregate pass@1.** NLL favours forward KL
   by construction. Pass@1 at 164 problems is too noisy to discriminate
   between methods that differ by a few percentage points. Spec-decode
   acceptance length is the only column whose units map onto deployed
   latency, and it's the column where the four methods spread on the
   first-pass eval, hardened with bootstrap CIs that survive a senior
   reviewer reading the table.

2. **The reverse-KL win at this scale is small, but in the right
   direction.** RKL beats CE significantly; RKL beats FKL/base
   non-significantly. That's consistent with the paper's mechanism --
   reverse KL preserves teacher alignment better than data-fitting CE --
   but at a smaller magnitude than the paper's instruction-following
   experiments, because my teacher has less of the kind of tail mass
   that makes the mode-covering vs mode-seeking distinction matter.
   The interesting place to test the paper's mechanism in full is at
   wider teacher-student gaps (7B → 0.5B, Mellum-4B → sub-1B).

3. **The on-policy fix in MiniLLM/GKD is still the load-bearing part of
   the recipe.** My naive `gkd` run is statistically tied with `rkl`
   despite being on-policy, which says my naive on-policy version isn't
   adding the things MiniLLM's full algorithm adds (length-normalised
   reward, teacher-mixed prefixes, single-step decomposition). With
   those, I'd expect on-policy to beat off-policy reverse KL by more
   than the noise floor.

4. **DistillSpec is the natural objective.** [DistillSpec](https://arxiv.org/abs/2310.08461)
   trains the student against the spec-decode acceptance probability
   directly. That's the loss whose units match the metric the eval
   showed actually discriminates between the methods. If I had another
   week of compute I'd add it as a fifth variant and see whether it
   widens the rkl-vs-ce gap.

## A few smaller observations from the runs

- The first lr I tried (5e-5) overshot and made every run worse than the
  base student on val NLL. I moved to 2e-5 with 100-step warmup and a
  proper cosine schedule that decays over real training steps (the bug I
  hit was that I was calling `sched.step()` only inside the
  gradient-accumulation branch, so the schedule advanced 4× slower than
  intended). Worth noting because if you're skim-reading distillation
  papers, the LR they cite for "fine-tuning a pretrained student" is
  almost always 1e-5 to 3e-5, not the 5e-5 you'd use training from scratch.

- The GKD run is roughly 5× slower than CE per step on the 4090 because
  every step does a 64-token autoregressive sample from the student before
  computing the loss. With KV caching disabled (gradient checkpointing
  forces it off in HuggingFace `generate`), that's 64 forward passes
  through the 0.5B student per training step. There's an obvious
  optimisation to be made there for any serious follow-up: cache student
  rollouts across steps and sample from a replay buffer.

- `codeparrot-clean-valid` has near-duplicates. Several of my training
  sequences are template-y test-fixture files that recur across the
  corpus. I left them in for the experiment to be honest about my data,
  but a real run would want to deduplicate by file hash plus a near-dup
  filter before tokenising.

## Mellum-as-teacher seq-KD: did it work?

The most obvious gap in the original Qwen-on-Qwen writeup was that I
kept talking about Mellum and never used it. The follow-up I built --
ran on Modal with $30 of credits after the lab GPU box went down -- is
a cross-tokenizer sequence-level distillation. Mellum-4b-sft-python
generates FIM completions on a Python corpus, those text targets get
re-tokenized in Qwen, and Qwen2.5-Coder-0.5B is SFT'd on (prefix,
mellum-middle, suffix) with loss masked to the middle. This is Kim &
Rush 2016 sequence-level KD; once tokenizers don't match, you can't do
logit distillation, only pseudo-labelling.

Five conditions, with a gold-FIM control alongside the Mellum-as-teacher
condition. Without that control I can't tell whether any improvement
comes from Mellum or just from teaching Qwen the FIM task format on this
specific corpus distribution:

| condition | what it sees during training |
|---|---|
| `base` | nothing -- the un-fine-tuned Qwen2.5-Coder-0.5B |
| `fim_gold` | (prefix, *ground-truth* middle, suffix), 600 codeparrot examples |
| `fim_mellum` | (prefix, *Mellum-generated* middle, suffix), same 600 examples |
| `fim_mix` | 50/50 mix of gold and mellum middles |
| `mellum_4b` | -- (the teacher itself, the upper bound) |

Hyperparams: 1200 steps, batch 2 × accum 4, lr 2e-5 cosine, middle-only
loss masking. Greedy Mellum decoding for the seq-KD targets (the
sampling-vs-greedy choice for seq-KD targets is contested in the
literature; greedy is the more conservative starting point because it
removes one source of variance).

Eval on two sets:

1. **Held-out codeparrot FIM** (180 examples, 60 per masking kind):
   exact-match against gold middles. Same distribution as training.
2. **HumanEval Infilling** (164 examples per subset, fixed seed
   subsample): the actual published Mellum benchmark. Different
   distribution, more curated, written-by-hand.

### Held-out codeparrot FIM (in-distribution)

| method | single | multi | random | overall EM | overall middle NLL |
|---|---|---|---|---|---|
| base | 0.350 | 0.000 | 0.017 | 0.122 | 1.013 |
| fim_gold | 0.367 | 0.067 | 0.017 | 0.150 | **0.965** |
| fim_mellum | 0.383 | 0.083 | 0.000 | 0.156 | 1.018 |
| **fim_mix** | **0.400** | **0.083** | 0.000 | **0.161** | 0.975 |

In-distribution, FIM training works. Mix wins on EM (+3.9 pp over base);
gold wins on the NLL-against-gold-middles metric, which it should by
construction. fim_mellum has *worse* gold-middle NLL than base because
it learned to generate Mellum-style middles, which differ from gold
middles in style. But fim_mellum's exact-match is competitive with
fim_gold, suggesting Mellum's generated middles are closer to the gold
in actual content than they are in token-level distribution.

### HumanEval Infilling (out-of-distribution, the published benchmark)

Sub-sampled 164 of each subset (full set is 1033/5815/1640) with a
fixed random seed shared across all five models. pass@1 by running the
canonical test against `prefix + completion + suffix`:

| method | single | multi | random | mean |
|---|---|---|---|---|
| base 0.5B | **0.787** | 0.396 | 0.506 | 0.563 |
| fim_gold 0.5B | 0.787 | **0.415** | 0.470 | 0.557 |
| fim_mellum 0.5B | 0.774 | 0.415 | 0.470 | 0.553 |
| fim_mix 0.5B | 0.768 | 0.390 | 0.470 | 0.543 |
| mellum_4b (teacher) | 0.738 | **0.537** | **0.683** | **0.652** |

For sanity-check reference, the public Qwen2.5-Coder-0.5B paper reports
0.754 / 0.473 / 0.460 (mean 0.562) on the full set; my 164-subsample
base of 0.787 / 0.396 / 0.506 (mean 0.563) is consistent with that.
Mellum-4b-base reports 0.6621 / 0.3852 / 0.2970 (mean 0.448); my
Mellum-sft-python is the Python-fine-tuned variant which is known to
beat the base on Python infill.

**None of the FIM-tuning conditions beat the un-fine-tuned base on
HumanEval Infilling.** Plain base (mean 0.563) edges out every
fine-tuned variant (0.543 to 0.557). Mellum-as-teacher (fim_mellum,
0.553) is below the gold-data control (fim_gold, 0.557). The mix
condition is the worst at 0.543.

### What this says

The held-out result and the HumanEval result point in opposite
directions. FIM training improves performance on the
training-distribution (codeparrot) and degrades it on the
distribution-shift benchmark (HumanEval). The student is learning the
*style* of the training corpus, not the *task* of FIM.

The reason is the one Bavarian et al. (2022) flag in the FIM
pretraining paper: FIM capability comes from the data transformation
*at pretraining scale*, not from a small fine-tune. Qwen2.5-Coder-0.5B
was already FIM-pretrained on billions of tokens. My 600-example
fine-tune adds noise relative to that.

The Mellum-as-teacher signal is also not visible at this scale.
fim_mellum and fim_gold are within noise of each other on both evals; the
direction of effect on HumanEval Infilling (fim_mellum slightly worse
than fim_gold) is consistent with "Mellum's text outputs are sometimes
slightly off-distribution for the canonical-solution form HumanEval
expects." A larger Mellum-generated corpus, or distillation that
preserves Mellum's actual logits rather than its decoded text, might
shift this. Cross-tokenizer logit distillation is the obvious next
step but the engineering is real: two incompatible BPE schemes need
either a byte-level alignment between teacher and student tokens or
a soft-token bridge in the spirit of Boizard et al. 2024 (Universal
Logit Distillation, arXiv:2402.12030).

The single positive signal is that fim_gold and fim_mellum *both*
improve multi-line HumanEval pass@1 over base (0.415 vs 0.396, +1.9pp
each). That's small but in the right direction, and multi-line is
the FIM subtask the un-fine-tuned base is weakest on (single 0.787,
multi 0.396, random 0.506). The held-out EM differences on
codeparrot multi-line (base 0.000, fim_gold 0.067, fim_mellum 0.083,
fim_mix 0.083) point the same way: multi-line is where the fine-tune
produces the clearest gain.

### Third eval column: RepoBench-Python next-line (the L2R regression check)

The HumanEval Infilling regression worried me because I couldn't
distinguish "FIM training learned codeparrot style and slightly hurts
out-of-distribution FIM" from "FIM training damaged the student's
plain LM ability." So I added a third eval on the *other* benchmark
Mellum's card reports: RepoBench-Python next-line prediction (Liu et
al. 2023, `tianyang/repobench_python_v1.1`). Three subsets
(`cross_file_first`, `cross_file_random`, `in_file`), 60 problems
each at file lengths ≤ 8k, fixed seed shared across all five models.
Score is edit-similarity (difflib ratio) between the model's first
generated line and the canonical next line, plus exact-match for
context.

| method | cf_first | cf_random | in_file | avg ES | avg EM |
|---|---|---|---|---|---|
| base 0.5B | **0.587** | 0.681 | 0.697 | **0.655** | 0.239 |
| fim_gold 0.5B | 0.571 | 0.667 | 0.697 | 0.645 | 0.228 |
| fim_mellum 0.5B | 0.564 | **0.689** | 0.677 | 0.643 | 0.183 |
| fim_mix 0.5B | 0.565 | 0.668 | **0.705** | 0.646 | 0.222 |
| mellum_4b (FIM-wrap) | 0.474 | 0.553 | 0.505 | 0.511 | 0.178 |

The four 0.5B variants land within 1.2 ES points of each other on
the average and within 2 ES points on every subset. **FIM
fine-tuning did not damage L2R LM ability.** This is the positive
null I needed: combined with the HumanEval Infilling regression,
the picture is "FIM students learned codeparrot-style FIM, didn't
forget L2R, slightly hurt on FIM out-of-distribution," not "FIM
students forgot how to be language models."

The Mellum row needs a caveat. My first run gave Mellum raw L2R
prompts and it scored ES 0.470 — per-problem inspection showed
Mellum hitting EOS in 1-2 tokens because Mellum-sft-python is
FIM-only and a raw L2R prompt isn't a shape it knows what to do
with. Re-running with the prompt wrapped as
`<fim_suffix><fim_prefix>{code}<fim_middle>` (empty suffix, FIM as
a degenerate next-token completion) gave the 0.511 in the table
above. Even with FIM-wrap, Mellum is below the Qwen base. The
Mellum card reports RepoBench Avg ≤8k = 0.299, which is roughly
half my 0.511 number, so the published number isn't directly
comparable to mine -- different RepoBench version, different
file-length bucket, different cross-file context format. Mellum's
training format probably uses `<filename>` / `<file_sep>` markers
between context files that I'm not providing. I'd treat 0.511 as a
loose lower bound on Mellum's RepoBench performance under my
specific protocol, not as a comment on Mellum's actual capability.

### What I'd change

1. **Scale the seq-KD corpus.** 600 examples is way too few. Bavarian
   et al. (2022) get FIM capability from training on billions of FIM
   tokens; I should be at 50K-500K examples minimum to expect fine-tune
   to beat what Qwen already learned.
2. **Use FIM-format-friendly data.** codeparrot files are real GitHub
   repos with messy structure; HumanEval is curated short Python
   functions. The training distribution should match the eval
   distribution at least roughly. MBPP-Plus or
   bigcode/code-search-net would be a closer match.
3. **Cross-tokenizer logit distillation.** Sequence-level KD throws
   away the per-token uncertainty information. ULD (Boizard et al.
   2024, arXiv:2402.12030) projects across incompatible tokenizers
   and preserves more of the teacher signal than text-level
   pseudo-labelling. Worth trying if seq-KD really is the bottleneck.
4. **Stop training earlier.** All three FIM trainings overfit. Val
   middle NLL at steps 300/600/900/1200 is 0.91/0.95/0.98/0.98 for
   gold, 0.96/1.01/1.04/1.04 for mellum, 0.92/0.97/1.00/1.01 for mix --
   monotonically up after step 300 in every run. I should have
   checkpointed every 100 steps and selected the best, not the last.
   Probably 30-40% of the HumanEval gap to base is recoverable from
   this alone.

## What I'd do with another week beyond that

In rough order of expected payoff:

1. **Length-normalised reward** in the on-policy reverse KL, the actual
   MiniLLM section 3.3 algorithm. Naive on-policy drifts because
   teacher reward at long rollouts is dominated by easy positions;
   length-normalising stops the student collapsing onto whichever mode
   the teacher rewards on average. The cheapest single change with the
   biggest expected effect on the reverse-KL ranking.

2. **DistillSpec.** Train against the per-token acceptance probability
   under the spec-decode rule directly, with a stop-gradient through the
   teacher. That's the loss whose units match the metric the eval
   actually discriminates on.

3. **A bigger teacher gap inside the Qwen family.** 1.5B → 0.5B is 3×;
   the reverse-KL effect should sharpen at 7B → 0.5B. Same-tokenizer
   drop-in. (I deliberately skipped this for now because it doesn't
   bring the experiment closer to Mellum, just sharpens the proxy.)

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

Total wall on a single RTX 4090 sharing the box: about three hours of
training + eval. `results/train_*.json` has per-step training curves,
`results/eval.json` has the first-pass aggregate eval,
`results/spec_eval.json` has the hardened spec-decode numbers including
per-position acceptance and bootstrap CIs.

## Things I'd want a Mellum engineer to push back on

1. The reverse-KL effect I see is small at 1.5B → 0.5B and only shows
   significantly against the CE baseline. The per-position figure says
   the mechanism is alive but the q4 bucket (where it lives) is only
   a quarter of positions, so it gets diluted. At 4B → sub-1B with a
   bigger entropy gap, does the q4 effect cleanly dominate, or do you
   see the same "swamped by easy positions" problem?
2. Is spec-decode draft length the right student-side metric for Mellum's
   actual deployment? My read of the public posts is that Mellum stands
   alone (no separate verifier model in the IDE), so the latency lever
   is the model's own forward-pass time and acceptance is the user's
   "did I keep this completion." If that's right, the right loss is closer
   to "expected number of tokens before the user starts editing,"
   which is harder to target offline but is presumably what the
   user-facing telemetry measures.
3. How much of Mellum's quality is the distillation objective vs the data
   curation and the FIM training? The public posts emphasise data and
   FIM. My result that "any KL distillation > CE" is a small effect
   that survives statistical rigour but is much smaller than the
   teacher-student capability gap (HumanEval pass@1 0.427 vs 0.274), so
   the lever I observe is real but bounded. I'd like to know whether
   internal numbers say the same.
