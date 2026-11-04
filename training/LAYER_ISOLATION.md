# Stage 22 — isolated layers and two-seed conversations

## Fixed-history comparison

This follows [stage 21](WEIGHT_EVALUATION.md) on the same pinned `lr3 / 188`
checkpoint. The primary remains the six-language stage-7 checkpoint. No SFT,
DPO or full-model export occurs here.

The first specification SHA-256 is
`9d3bdb696d44a99a3715bc887c788dd97ce2baa4a031aefa5dc188cbe48dd71c`.
Seven variants compare the unchanged parent, layer-15-only strengths 3/6,
layer-19-only strengths 3/6, and a strength-6 norm-matched orthogonal control
for each layer. Layer 15 stays byte-identical in layer-19 interventions and
vice versa. All other parameters are covered by the unchanged-parameter digest.

The nine consumed development cases now receive two seeds. Fixed histories use
their original turn seed and that seed plus 95; native-open-prefix questions use
42/137. These are paired seed slots, not pooled samples with the same numeric
turn seed. Sampling remains temperature 0.8, top-p 1, top-k 20, presence penalty
0.8, repetition penalty 1, with 256 new tokens maximum.

All **126 generations and 21 paired contrasts** completed on an A40 with BF16,
SDPA, Transformers 5.22.1 and reference PyTorch kernels. All 126 reached EOS.
All nine original-seed unchanged answers reproduce stage 21 token for token.
Both edited tensors restore exactly and every other parameter's digest matches.

Each Hebrew cell below contains four fixed histories. The second seed produces
no exact copy even in the unchanged parent.

| Variant | Exact copies, seed slot 0 / 1 | Mean previous-answer 4-word overlap, slot 0 / 1 | English final names retained, slot 0 / 1 |
| --- | --- | --- | --- |
| Unchanged | 1/4 · 0/4 | 36.15% · 3.33% | 3/3 · 2/3 |
| Layer 15, strength 3 | 0/4 · 0/4 | 2.42% · 1.61% | 3/3 · 3/3 |
| Layer 15, strength 6 | 0/4 · 0/4 | 3.33% · 0.00% | 2/3 · 3/3 |
| Layer 15, orthogonal 6 | 0/4 · 0/4 | 7.26% · 0.66% | 2/3 · 2/3 |
| Layer 19, strength 3 | 0/4 · 0/4 | 9.93% · 0.00% | 3/3 · 3/3 |
| Layer 19, strength 6 | 0/4 · 0/4 | 22.15% · 12.50% | 3/3 · 3/3 |
| Layer 19, orthogonal 6 | 1/4 · 0/4 | 35.30% · 0.00% | 3/3 · 2/3 |

Layer-15 isolation changes the earlier control result: its strong orthogonal
control also breaks the copied answer. Layer 19 retains the directed/control
difference on that original-seed case. Every variant closes the native thinking
block immediately on all six open-prefix probes. The Hebrew fixed-history
answers have no mixed Latin/Hebrew words or CJK spans in either seed slot;
ordinary morphology and scene contradictions remain visible.

The teacher-forced productive-minus-repeat margins move differently from the
free generation result. Layer 15/3 improves the three margins by +0.0202,
+0.0423 and +0.0332 nats/token. Layer 19/3 changes them by −0.0175, −0.0254
and −0.0314. These scores compare two fixed complete continuations with
different lengths; the generated replacement is a third continuation.
Full token-level likelihoods and projections are retained privately.

## Fresh dialogue comparison

After this screen, freeze **layer 19 / strength 3** for the follow-up: it breaks
the copied answer, retains all three English final names in both seeds, and
has lower Hebrew overlap than strength 6. The follow-up adds a new
**layer-19 / strength-3 orthogonal control**, matching the chosen magnitude.

Fresh scenes were authored before candidate selection; their SHA-256 is
`a0820d95cabed8e5c8e0db87503828e1808f88d254bbf70285732659d5e728f7`.
The frozen follow-up specification SHA-256 is
`8fb92816d5f718732ac99f8c57d8abf2f79f43665a18eb381d871a86ede7d96e`.
Each variant receives 16-turn English, Russian and Hebrew conversations with
seeds 42/137: 288 fresh answers. Topics change repeatedly; object ownership and
custody are revisited after loans, returns and gifts; a female first-person
memory interrupts the dialogue; three distinct final actions must close it.

Every variant/seed starts with an empty conversation and extends only its own
actual generated answers. Full messages and turn seeds are saved. The old nine
fixed probes and three contrasts also rerun for all three variants, supplying
54 fixed generations and nine contrast pairs. The shared generation function
keeps fixed and conversational decoding identical.

### Completed follow-up

All **342 generations** reached EOS, including all 288 conversational turns.
The 36 fixed generations shared with phase A reproduce token for token after
the dialogue refactor. Both edited tensors restore exactly; the unchanged
parameter digest is identical before and after. The runtime pass took 751
seconds after model loading. The rounded control/target perturbation-norm ratio
is **1.000029**; layer 15 stays unchanged in both interventions.

On the old Hebrew history, the magnitude-3 control preserves the exact copied
answer at the original seed, while the directed magnitude-3 edit removes it.
Across the fresh dialogues, **every variant has zero whole-answer copies**.
Mean overlap with earlier assistant answers is:

| Variant | EN, seeds 42 / 137 | RU, seeds 42 / 137 | HE, seeds 42 / 137 |
| --- | --- | --- | --- |
| Unchanged | 1.32% · 1.63% | 6.87% · 0.17% | 3.14% · 2.02% |
| Layer 19, strength 3 | 1.17% · 6.88% | 1.04% · 6.15% | 9.62% · 4.09% |
| Layer 19, orthogonal 3 | 2.92% · 1.23% | 7.41% · 3.11% | 2.46% · 0.35% |

Each cell contains 16 turns, with overlap measured only against that branch's
actual previous answers. Some turns deliberately return to earlier objects.
The directed edit increases Hebrew phrase reuse in both seeds on these fresh
scenes. No mixed-script words or CJK spans appear in the 288 dialogue answers.

Reading the saved histories finds the same object-state failures: a gifted
object is still assigned to its donor, and an earlier borrower remains the
reported holder after a return. This happens in English too. The directed
variant has a correct two-part ownership answer on English seed 137, turn 12;
the surrounding Russian/Hebrew answers still confuse the state. Its Russian
female-memory requests turn into third-person narration in both seeds.
English turn 15 retains the three requested closing actions at seed 137 but
loses one participant at seed 42; the unchanged parent has the same split.

The Russian literal counter finds one `подала` answer in the directed variant
and one in the control, both at seed 42. Reading distinguishes them: the
directed answer has mismatched subject agreement; the control describes a woman
handing over an envelope, with matching agreement. The exact role-verb counter
returns zero, while direct reading catches a Hebrew performance-frame phrase
outside its current patterns on directed seed 137, turn 9. Those observations
are retained with source IDs rather than turned into an aggregate quality score.

**Keep the existing primary.** The isolated operation breaks the reproduced
old copy; this fresh screen supplies no whole-answer-copy advantage over the
parent and retains the language/state failures. No edited model is promoted.
The private blinded RU/HE bundle contains four conversation/seed groups,
three shuffled complete versions of each, and all 192 actual answers. Its
separate attribution key is retained for the external reading. No fresh visual
acceptance run or new SFT/DPO pass was performed in this stage.

## Corpus coverage checked before the next SFT

The stage-19 training data already contains 140 `he_ownership` and 140
`he_female` records. All 280 have empty history. The ownership records include
gifts, loans, safekeeping, named people, speaker-relative possession and
two-object questions. The next curriculum should extend these constructions
across actual dialogue histories with explicit owner/holder state after every
transfer, rather than treating the category as absent. Fresh evaluation scenes
from this stage remain outside training.

## Reproduction

```sh
python training/weight_probe.py --spec /private/spec.json \
  --target lr3-step188 --output /private/isolation.jsonl
python training/summarize_weight_probe.py --spec /private/spec.json \
  --input /private/isolation.jsonl --output /private/isolation-summary.json
python training/weight_probe.py --spec /private/fresh-spec.json \
  --target lr3-step188 --output /private/fresh.jsonl
python training/summarize_weight_probe.py --spec /private/fresh-spec.json \
  --input /private/fresh.jsonl --output /private/fresh-summary.json
```

The summary rejects incomplete or duplicate cells, failed restoration, changed
specification hashes, and dialogue histories borrowed from a different branch.
It reports each language and seed slot separately. Raw generations, private
specifications, source snapshots, control hashes and operator receipts stay in
the private HF dataset repository.

All 84 repository tests passed in the GPU environment. The local run passes
80 tests and skips four PyTorch-dependent checks. The new regression checks
cover branch-history isolation, seed offsets, missing and duplicate cells,
changed fixed prompts, mismatched installed hashes, and failed restoration.
