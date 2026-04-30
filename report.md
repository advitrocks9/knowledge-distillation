# Forward vs reverse KL when distilling a code model

I read MiniLLM (Gu et al., ICLR 2024) and wanted to know whether its
reverse-KL claim survives contact with a real code teacher and student. The
short answer, after running 1.5B → 0.5B Qwen2.5-Coder distillation on Python
source under four loss variants, is that the metric you grade on flips the
ranking, and the metric that matters most for an inference-bound team is
the one nobody in the literature reports.

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

| run | held-out NLL | HumanEval pass@1 | spec K=2 (max 2) | spec K=4 (max 4) |
|---|---|---|---|---|
| teacher Qwen2.5-Coder-1.5B | **1.0725** | **0.427** (70/164) | 2.000 | 3.984 |
| student Qwen2.5-Coder-0.5B (no fine-tune) | 1.2845 | 0.256 (42/164) | **1.640** | 2.396 |
| student + ce  | **1.2818** | **0.274** (45/164) | 1.528 | 2.355 |
| student + fkl | 1.2870 | 0.268 (44/164) | 1.387 | **2.518** |
| student + rkl | 1.3328 | 0.262 (43/164) | 1.576 | 2.390 |
| student + gkd | 1.3181 | 0.268 (44/164) | 1.492 | 2.446 |

Numbers in `results/eval.json`; per-problem HumanEval completions live
inside that file under `humaneval_completions`. Each spec-decode entry is
the mean of 32 prompts × multiple draft cycles (4 for K=2, 2 for K=4).

The table tells three different stories depending on which column you read:

**Held-out NLL.** CE wins narrowly, FKL is roughly tied with the un-fine-
tuned student, RKL and GKD make NLL strictly worse. This is forced by
maths: forward KL minimises `E_{x~p_T}[-log p_S(x)]` which is the
cross-entropy of `p_S` against the teacher's distribution, not the data's,
and the teacher and the data already largely agree at this scale. Reverse
KL minimises the cross-entropy of `p_T` against `p_S`, which is *not*
NLL, so it's expected to make NLL worse on the way to making something
else better. The NLL column is a sanity check, not a discriminator.

**HumanEval pass@1.** All distilled students beat the un-fine-tuned base
(0.256 → 0.262-0.274), but plain CE wins. CE picks up 6 problems the base
got wrong and loses 3, FKL picks up 4 and loses 2, RKL picks up 2 and
loses 1, GKD picks up 4 and loses 2. The CE / FKL gap on HumanEval is one
problem out of 164, which is barely outside coin-flip noise on this size of
benchmark. So the right reading is "all four objectives are roughly equal
on the user-facing metric" -- the experiment doesn't rank them here.

**Speculative-decoding draft acceptance, K=4.** Forward KL wins, by a
margin that's about 5% above the base student and about 7% above CE
(2.518 vs 2.355). This is the metric where the loss-shape effect is
visible, and it's visible *for forward KL*, not reverse. The mechanism is
the one MiniLLM warns against in reverse: at K=4 the student is asked to
match the teacher's distribution over four consecutive positions, and FKL
trains exactly that, while CE drifts the student toward the data's
empirical marginals (which the teacher has already smoothed past with its
extra capacity). Reverse-KL students don't beat FKL on K=4 because at this
teacher size (1.5B, not 70B) and this training duration (~10M tokens) there
isn't enough of a "harmful tail" in the teacher for mode-seeking to pay
off; they end up matching the teacher slightly worse than FKL because RKL
doesn't directly target distribution match.

The K=2 column inverts this: the un-fine-tuned student wins, and FKL is
*worst*. I read that as a Goodhart-style artefact of the per-position
ratios. The base student is sharpest where the local context is most
informative (the very first drafted token after a real prefix), so the
first per-token acceptance prob is high; FKL trains the student to spread
mass like the teacher, which lowers the first-token acceptance but
preserves the alignment over longer windows. K=4 / K=2 ratios bear this
out: base 1.46, FKL 1.82 -- FKL acceptance decays geometrically over the
draft, base front-loads.

## What this changed about my read of the paper

I came in expecting reverse KL on a real code teacher to clearly beat
forward KL on at least one metric, the way MiniLLM's instruction-following
results suggest. What I got was that forward KL won the metric closest to
what JetBrains actually ships, and reverse KL won nothing. Three things
this updates me on:

1. **The reverse-KL win is conditional on the teacher having structured
   tail mass.** MiniLLM's instruction-following teacher (a 13B chat model)
   has plenty of plausible-but-wrong continuations to ignore; my 1.5B code
   teacher has comparatively little, because both teacher and student were
   pretrained on the same large code corpus and largely agree at this
   scale. The interesting place to test reverse KL on code is a much wider
   teacher-student gap -- 7B teacher into 0.5B student, or Mellum-4b into
   a sub-1B student.

2. **The on-policy fix in GKD/MiniLLM is the load-bearing part of the
   recipe, not the reverse-KL switch by itself.** My naive `gkd` run
   sampled from the student and scored with the teacher; it didn't include
   length-normalised reward, didn't mix in teacher-sampled prefixes
   periodically (the standard fix for early-training collapse), didn't do
   the policy-gradient single-step decomposition. With that recipe missing,
   the on-policy run beats teacher-forced reverse KL on NLL (1.32 vs 1.33)
   but doesn't approach forward KL on the metric I care about. The lesson:
   "use reverse KL" is the headline; the real engineering is in the things
   nobody quotes from the paper.

3. **The right objective for an inference-bound team is the spec-decode
   acceptance length, not pass@1 or NLL.** This is the only column in the
   table where the loss-shape effect is unambiguous, and it's also the
   only column whose units map directly to user-perceived latency. If I
   were running a Mellum-style distillation experiment for real I would
   reward-shape against this directly, not against pass@1 (which is too
   small to differentiate at this number of problems and too downstream to
   pin to the loss) and not against NLL (which favours the wrong loss).
   [DistillSpec](https://arxiv.org/abs/2310.08461) (Zhou et al., 2023)
   does this explicitly; it should be the default reference for any
   distillation work targeted at speculative decoding.

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

## What I'd do with another week

In rough order of expected payoff:

1. **Length-normalised reward**, the actual MiniLLM section 3.3 algorithm.
   Naive on-policy drifts because teacher reward at long rollouts is
   dominated by easy positions; length-normalising stops the student
   collapsing onto whichever mode the teacher rewards on average. That's
   the cheapest single change with the biggest expected effect on the
   reverse-KL ranking.

2. **A bigger teacher gap.** 1.5B → 0.5B is 3×. The reverse-KL effect is
   sharper at 7B → 0.5B or larger. Qwen2.5-Coder-7B-Instruct as a teacher
   into the same 0.5B student is the same-tokenizer drop-in. About 14GB
   for the teacher, doable on a 4090 with the rest offloaded.

3. **Distil for spec-decode acceptance directly.** DistillSpec's actual
   loss is the per-token acceptance probability under the spec-decode
   rule, with a stop-gradient through the teacher. That's the loss whose
   units match the metric I'd actually grade on.

4. **FIM-aware distillation.** Mellum is FIM-trained
   (`<fim_prefix>`/`<fim_middle>`/`<fim_suffix>`) and code completion is
   inherently a FIM task. Standard distillation runs the same loss at
   every position; an obvious refinement is to weight the loss towards
   the `<fim_middle>` segment, which is what users actually accept. None
   of the public distillation papers I read run this ablation and it's
   the place where the JetBrains-specific signal probably lives.

## How to repro

```bash
uv sync
uv run python data.py                                                 # ~5s
uv run python distill.py --loss ce  --steps 2500 --lr 2e-5            # ~5 min
uv run python distill.py --loss fkl --steps 2500 --lr 2e-5            # ~7 min
uv run python distill.py --loss rkl --steps 2500 --lr 2e-5            # ~7 min
uv run python distill.py --loss gkd --steps 2500 --lr 2e-5            # ~40 min (sampling)
uv run python eval.py                                                 # ~50 min (HumanEval × 6)
uv run python make_table.py                                           # prints the markdown
```

Total wall on a single RTX 4090 sharing the box: about two hours of
training + eval. `results/train_*.json` has per-step training curves,
`results/eval.json` has every number behind every table.

## Things I'd want a Mellum engineer to push back on

1. Is the "forward KL wins K=4 spec-decode at 1.5B teacher" result still
   true at 4B? My intuition is yes -- nothing about the mechanism scales
   the wrong way -- but the gap to reverse KL might invert if the 4B
   teacher has the kind of structured tail mass MiniLLM's argument
   relies on.
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
   FIM; my experiment suggests the distillation lever exists but is
   smaller than a recipe choice (CE vs FKL on HumanEval is one problem
   out of 164). I'd like to know which side of "distillation matters"
   the team's internal numbers fall on.
