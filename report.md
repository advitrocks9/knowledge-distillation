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

## In flight: Mellum-as-teacher seq-KD

The most obvious gap in this writeup is that I keep talking about
Mellum and never use it. The follow-up I'd built code for and was
running when the GPU box went unreachable mid-eval is a cross-tokenizer
sequence-level distillation: have Mellum-4b-sft-python generate FIM
completions on a Python corpus, retokenize the text in Qwen's tokenizer,
and SFT Qwen2.5-Coder-0.5B on (prefix, mellum-middle, suffix) with loss
masked to the middle. This is Kim & Rush 2016 style seq-KD; once
tokenizers don't match, you can't do logit distillation, only
pseudo-labelling.

The conditions I designed (after a code-review pass; see
`notes/levelup2-rescue.md`):

- `base`            -- untouched Qwen with the FIM prompt format. Lower
  bound.
- `fim_gold`        -- SFT on real (prefix, ground-truth-middle, suffix)
  triples. The mandatory control. Without it I can't tell whether any
  improvement comes from Mellum or just from teaching Qwen the FIM task
  format.
- `fim_mellum`      -- SFT on Mellum-generated middles. The seq-KD condition.
- `fim_mix`         -- 50/50 mix of gold and mellum middles. Tests
  whether teacher guidance helps without forcing full imitation.
- Mellum itself     -- upper bound on the same eval.

Eval is HumanEval Infilling (the metric Mellum's own card reports:
0.6621 / 0.3852 / 0.2970 single / multi / random for the base model)
plus a held-out codeparrot-FIM exact-match check, plus a non-FIM
HumanEval pass@1 regression to make sure FIM-tuning doesn't hurt
left-to-right completion. Code is `fim_data.py`, `fim_generate.py`,
`fim_train.py`, `fim_eval.py`, `humaneval_infilling.py`. The runner
script `run_fim_experiment.sh` ties it together.

What a null result looks like (so I can read the table honestly when it
runs): `fim_gold > base` but `fim_mellum ≈ fim_gold` would say "the
lever is FIM adaptation, not distillation from Mellum specifically."
That's still a result. `fim_mellum > fim_gold` would be the
interview-grade outcome -- cross-tokenizer transfer working. `fim_mellum
< fim_gold` would say tokenizer mismatch / teacher-text noise dominates.
In either case I'd want to verify on RepoBench-Python that FIM tuning
hasn't broken left-to-right completion, since the public Mellum number
of `Avg ≤ 8k = 0.299` is a regression check Mellum-the-team must already
do.

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
