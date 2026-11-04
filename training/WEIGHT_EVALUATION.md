# Stage 21 — controlled weight-edit inference

## Connector finding

PR 25's inline P2 report identified a real omission in the English role-frame
counter: mandatory whitespace after `I` matched `I am portraying`, but skipped
`I'm portraying`. The counter now accepts both straight and typographic
apostrophes, preserves literal spans, and checks word boundaries. Regression
tests cover contractions, the uncontracted form, case, whitespace and negatives.

Recounting all 298 English stage-19 generations changed zero observations
(zero before and after). The earlier RU/HE attribution is unchanged.

## Frozen comparison

This is the inference follow-up to [stage 20](WEIGHT_DIAGNOSTICS.md). No SFT or
DPO updates occur. The two checkpoints are:

| Checkpoint | Private HF revision | Weight-file SHA-256 |
| --- | --- | --- |
| Primary, six-language epoch one | `805bc6e493f284be8b20c262b8fe65070c886c27` | `e365806211389de6513e1e7c7706e54b2203531d8e16acceb843488a0f707d5b` |
| Stage 19, lr3 / update 188 | `04e78daa7114ffbce835552a2db52119c91e48e7` | `45d9d3d8f2762841d87876aa6d63f64fa33327b10909ebf4736d2ea412db750b` |

Each receives seven variants: unchanged, targeted strengths 0.25/1/3/6, and
orthogonal controls matched to strengths 1/6. Both layers 15 and 19 are edited
together. The direction, 64 columns per layer, control seeds and candidate
tensor hashes were frozen in stage 20. Each variant starts from the same parent;
the target's own column coefficients are used. CPU-prepared and installed BF16
bytes must match exactly. Unsupported columns are checked before installation.

The private specification SHA-256 is
`125618c75153ca01082437d10f0baab9171e20f34a1585ba18bb52fb6809f406`.
Its scenes are consumed stage-19 development cases. For each variant:

- Three saved Hebrew histories contrast an observed exact previous-answer copy
  with an Astra-authored continuation grounded in the original scene rule.
  Both branches receive the same prefix. Raw answer-token log probabilities,
  the first-16-token score and per-token MLP projections are retained.
- Six fixed-history generations cover those three situations and the final
  English/Russian/Hebrew turns of the primary's saved long conversations.
  Each receives its original saved `turn_seed` (55, 56 or 57).
- Three native open-thinking-prefix probes use the prior RU/HE/EN questions,
  seed 42. Thinking text, the generated closing tag and EOS are retained.

Generation uses temperature 0.8, top-p 1, top-k 20, presence penalty 0.8,
repetition penalty 1 and a 256-token cap. Likelihood scoring applies no sampling
penalties. Productive-minus-repeat differences use mean token log probability;
the two branches have different lengths, and all token-level values are saved.
No history or continuation is silently truncated.

Execution uses an A40, Transformers 5.22.1, BF16, SDPA and the reference PyTorch
linear-attention/convolution implementations, matching the previous experiment.
The primary's three original final replies reproduce token for token.

## Results

Both comparisons completed: **126 generations, 42 paired contrasts / 84
teacher-forced continuations**. The two runtime passes took 176 and 147 seconds
after model loading. Restoring the edited tensors succeeded; the digest of all
other parameters stayed identical for both checkpoints.

### Copying on the four Hebrew fixed histories

These are four development histories per cell, including one reproduced exact
copy in the unmodified lr3/188 model. The other checkpoint's histories are
deliberately supplied unchanged, rather than regenerated separately per variant.

| Variant | Primary: exact copies | lr3/188: exact copies | lr3/188: mean prior-answer 4-word overlap | lr3/188: Latin/Hebrew mixed-word answers |
| --- | ---: | ---: | ---: | ---: |
| Unchanged | 0/4 | 1/4 | 36.15% | 0/4 |
| Targeted 0.25 | 0/4 | 1/4 | 34.38% | 0/4 |
| Targeted 1 | 0/4 | 1/4 | 34.00% | 0/4 |
| Targeted 3 | 0/4 | 0/4 | 6.82% | 1/4 |
| Targeted 6 | 0/4 | 0/4 | 6.25% | 0/4 |
| Orthogonal control 1 | 0/4 | 1/4 | 35.23% | 0/4 |
| Orthogonal control 6 | 0/4 | 1/4 | 31.25% | 0/4 |

The directed intervention eliminates the reproduced exact copy at strengths
3/6; both orthogonal controls retain it. Reading the replacements still finds
ordinary morphology failures and a scene contradiction: a prompt about a cloud
that already rained receives a description of a cloud without rain. The targeted
3 variant also introduces a mixed Latin/Hebrew word in a separate final reply.

On the primary's English final turn, strengths 1/3/6 reduce the answer from
58 to 24 tokens and omit the distinct final actions of two named participants.
The original and strength 0.25 preserve all three; the stronger orthogonal
control also retains all three names. On lr3/188, the targeted 6 answer reopens
the bakery after closing it, contrary to the requested ending.

Native empty thinking closure is unchanged: **lr3/188 closes 3/3 probes under
every variant**, while the primary closes 0/3 under every variant. The primary's
EOS inside an unclosed thinking block remains separately recorded.

### Measured layer writes and continuation likelihood

For each teacher-forced continuation, the hook measures the selected columns'
contribution along the published direction at the positions predicting answer
tokens. The RMS ratio of repeat-branch to productive-branch contributions is:

| Checkpoint | Layer 15, range across three pairs | Layer 19, range across three pairs |
| --- | ---: | ---: |
| Primary | 3.33–4.47× | 1.12–1.52× |
| lr3/188 | 3.51–4.12× | 0.95–1.08× |

For lr3/188, targeted strength 6 changes the productive-minus-repeat mean
log-probability margin by +0.0192, +0.0152 and +0.0231 nats/token on the three
pairs. The norm-matched control changes it by +0.0042, +0.0126 and +0.0029.
The raw repeat remains more probable on all pairs. These are BF16 forward-pass
measurements with FP32 log-softmax; there is no FP32 model rerun in this stage.
Token-level scores and projections are retained for such a follow-up.

## Selection and prepared follow-up

**Keep the existing primary.** The combined edit earns a narrower layer
comparison through its copying result; no variant is promoted from this
development screen. No fresh acceptance or photographic recheck was run.

Phase B is prepared and hash-pinned: unchanged lr3/188, layer-15-only and
layer-19-only strengths 3/6, plus a norm-matched strength-6 control for each
layer. It retains the same nine diagnostic generation cases and three contrast
pairs. Its specification SHA-256 is
`5613d5bc278be7ce851b61a9c8fe770fa81cd51faaa93efe9f01fd6b09cb1e0a`.
Layer isolation is **not executed**: after the completed run's automatic stop,
RunPod rejected resumption and new A40, RTX 4090 and RTX A6000 placements for
insufficient available capacity. API receipts are retained privately.

Next, run that layer comparison with a reproduced unchanged baseline, then
freeze any candidate before fresh RU/HE/EN topic-switch conversations and visual
checks. The separate owner/holder and Hebrew feminine-language curriculum from
stage 20 remains the next data intervention.

## Reproduction

`weight_probe.py` verifies full checkpoint files, target tensors and every
candidate against the private specification. It records actual inputs, generated
IDs, literal error spans, previous-answer overlap and per-token activation
projections. The output's final integrity record verifies restoration of both
edited tensors and the digest of every other parameter, including language
interfaces and the visual encoder.

```sh
python training/weight_probe.py --spec /private/spec.json \
  --target selected --output /private/selected.jsonl
python training/weight_probe.py --spec /private/spec.json \
  --target lr3-step188 --output /private/lr3-step188.jsonl
```

All 78 phase-A repository tests passed on the GPU environment. The final local
suite includes the new layer-isolation test: 75 passed and four PyTorch-dependent
tests were skipped. Raw character generations, contrasts,
activation arrays, the private specification and operator receipts stay in the
private HF dataset repository.
