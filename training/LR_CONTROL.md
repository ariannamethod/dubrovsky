# Russian/Hebrew learning-rate control, stage 19

Both matched SFT passes are complete. Increasing peak LR to `3e-6` lowers
validation loss and raises the final Russian automatic grammar score on both
generation seeds. Owner/holder failures, Hebrew morphology and repeated whole
answers in long Hebrew conversations remain. The stage-7 primary is retained;
`1e-6 / update 80` and `3e-6 / update 188` enter the new external blind packet.

Stage 19 compares peak learning rates `1e-6` and `3e-6` from the same pinned
stage-12 checkpoint. The stage-18 training and validation files are copied byte
for byte. Both arms use seed 20261104, one epoch, accumulation 16, 188 optimizer
updates, the same unchanged trainer, and the same frozen embedding/head/visual
interfaces. No new training examples are introduced in this comparison.

The previous L40S pod could not resume because its host had no free GPU, and
new Secure Cloud L40S allocation also failed. Both arms therefore run on one
A40. Repeating the `1e-6` arm on that A40 provides a matched hardware control;
the historical L40S outputs remain a separate reproducibility comparison.

## External stage-18 review

Don/Fable read fourteen RU/HE questions with three anonymously shuffled answers
each. Decoding his stated choices gives:

| Language | Stage-7 primary | Stage-18 update 80 | Stage-18 update 188 | None |
| --- | ---: | ---: | ---: | ---: |
| Russian, seven questions | 1 | 4 | 1 | 1 |
| Hebrew, seven questions | 2 | 1 | 1 | 3 |

His highest-rated Russian scene, 01B, comes from update 188. The female narrators
he accepted in 01B, 01C and 02A come from update 188, update 80 and update 80.
His preferred Hebrew line about adding dependencies, 14B, comes from the
stage-7 primary. These are choices between literal saved outputs; the raw review,
blind key and source hashes remain private.

Attribution also separates the failure classes:

| Observed item | Stage-7 primary | Update 80 | Update 188 |
| --- | ---: | ---: | ---: |
| Mixed Latin/Cyrillic word in seven RU answers | 1 | 0 | 0 |
| `подал/подала/подало/подали` in seven RU answers | 1 | 0 | 1 |
| Initial `שנית/שניתן` in seven HE answers | 0 | 3 | 3 |

The exact mixed word `инсекtes` belongs to the primary. The two feminine
`подала` occurrences belong to the primary and update 188. The Hebrew opening
occurs six times among the packet's 21 Hebrew answers (the original review
reported a denominator of 18). On the complete sixteen-question HE set, the
stage-12 parent and stage-7 primary each have one such opening, update 80 has
four and update 188 has three. The subset and complete-set denominators remain
separate in the receipts.

For question 11B, Fable accepts the intended owner/holder relation and assigns
condition score 2. The answer explicitly names Yael as owner but says
`אצל איתי` for the holder; Astra's stage-18 whole-answer check did not accept it
as a complete owner/holder success. Both readings are retained. Stage 19 records
owner, current holder, grammatical form and surrounding contradictions
separately so an intended relation and a malformed surface form remain visible.

## Fresh evaluation and literal counters

Before generation, 42 new prompts were frozen: 12 RU, 16 HE, eight EN, and two
each in FR/DE/ES. They include four owner/holder questions in each target
language, three female-narrator questions, action order, agreement and ordinary
conversation. Three new sixteen-turn RU/HE/EN conversations test topic changes,
conditional object behavior, loans, gifts, returns and delayed recall. Each
conversation uses that model's actual preceding answers. Questions do not match
training/validation questions or user turns in their histories.

Stage-18 questions remain consumed diagnostics. The next external blind packet
is drawn from the new scenes. Native open/closed thinking probes and photographic
probes remain regression checks. Text sampling settings are unchanged: temperature
0.8, top-p 1, top-k 20, presence penalty 0.8, repetition penalty 1 and maximum
512 new tokens, with native end tokens. Fresh cases use both seeds 42 and 137;
dialogue/thinking/photo checks use seed 42. Seed 42 is fixed for the external
blind packet. Diagnostics keep seeds and sampling configurations separate.
Photographic probes retain the separate `qwen-vision` preset: temperature 0.7,
top-p 0.8, top-k 20, presence penalty 1.5 and at most 192 new tokens, seed 42.

`text_diagnostics.py` records exact mixed-script word spans, all four Russian
past-tense `подал` forms, initial Hebrew `שנית/שניתן`, leading token pairs and
case-folded prompt word overlap with literal answer evidence. Separate English words and hyphenated compounds do not
count as a mixed-script word. Prompt-overlap counters distinguish the opening
from the full answer; direct reading determines whether an overlap is an echo
or a useful restatement. Counts retain checkpoint/language denominators.

Example invocation on private generation files:

```bash
python training/text_diagnostics.py results/lr19/evaluation/*-fresh.jsonl \
  --output results/lr19/fresh-diagnostics.json
```

## Review protocol correction

Direct reading of the first combined API rubric found two evaluator errors:
it sometimes assigned the assistant's first-person possession to the user,
and it missed ordinary morphology while filling its evidence budget with
scene contradictions. The original requests, responses and scores are retained.

Every checkpoint receives the same additional grammar-only pass, in anonymous
batches of at most three answers. Ownership questions receive a separate
extraction of each object's owner and current-holder claims, with exact answer
quotations. Code compares the extracted actor labels against the pinned scene
facts. Missing claims, ambiguous forms and explicit contradictions retain
different labels. The original rubric supplies the scene/continuity and voice
axes; the separate passes supply the final automatic grammar and relation
columns. Both use the pinned API model `gpt-5.5-2026-04-23`; the separate passes
use medium reasoning effort. These are evaluator requests, independent of
Dubrovsky's empty-thinking training target.

Literal quotation checks reject invented or borrowed evidence and send those
items for isolated rereading. Direct reading also records residual evaluator
misses. Automatic scores remain identified as such, alongside the source
answers and the fresh external blind packet.

## Completed execution and integrity

Executed repository base: `59317f616e23346c872542af5c0735366c47ca87`.
The stage-12 parent is pinned in `ataeff/Dubrovsky` at revision
`a88b6374dbf496f3c4c6cea93004cdebefeb4251`, path
`working/joint-multilingual-empty-closure-stage12`, weight SHA-256
`74a27581a4cba0eff0170d5b373340c59791a37e8fd8e8d994687871b78758e1`.
Training SHA-256: `e5cdf07ae8367343b73807431f9f1aff3ed2219af2b50d7fb81d6ed938bc82ec`.
Validation SHA-256: `ba6c143723b7e08dcee80681525518957248692e14eadb9fdf71723249065c8f`.

Both arms start from the same pinned stage-12 parent and train 498,113,344
language-body parameters out of 852,985,920 total parameters. Each sees all
3,004 examples exactly once: 236,818 supervised tokens, 188 optimizer updates,
one epoch. The byte-identical sample order also matches stage 18. The training
mixture contains 489 EN, 1,164 RU, 1,303 HE, and sixteen each FR/DE/ES records.
Training uses FP32 parameters/Adam state and BF16 autocast; saved candidates
are BF16. The maximum complete training sequence is 2,356 tokens (limit 4,096).

There are eighteen verified nonzero private weight backups: eight from `1e-6`
and ten from `3e-6`, including both early comparison points. Every exported
embedding/head, all 153 visual tensors and tokenizer files match the parent.
Captured inventories of 45 executed source files and all input files match.
Initial validation is identical between the two A40 arms.

The battery contains 1,296 generations: 840 fresh answers, 180 thinking probes,
240 turns across five sets of three sixteen-turn conversations, and 36 image
answers. Exact own-model dialogue histories were verified. Each text arm has
one training seed; fresh generation seeds 42 and 137 are reported separately.

## Training

| Arm | Updates | Online token-weighted train loss | Best validation | Best update | Nonzero weight backups |
| --- | ---: | ---: | ---: | ---: | ---: |
| lr1 | 188 | 2.707543 | 2.553573 | 188 | 8 |
| lr3 | 188 | 2.620768 | 2.540839 | 188 | 10 |

## Fresh automatic ratings

Grammar comes from the separate language-only pass; scene and voice retain the original blinded rubric. Grammar/voice range 0–3; scene ranges 0–2. Each seed has 12 RU, 16 HE and eight EN answers. `selected` is stage 7, `parent` is stage 12; `stage18step*` are historical L40S-trained exports.

### RU

| Model | Grammar 42 / 137 | Scene 42 / 137 | Voice 42 / 137 | Both relations correct 42 / 137 |
| --- | --- | --- | --- | --- |
| selected | 1.417 / 1.750 | 0.667 / 0.667 | 1.583 / 1.667 | 0/4 / 0/4 |
| parent | 1.750 / 2.000 | 0.583 / 0.750 | 1.500 / 1.833 | 0/4 / 0/4 |
| lr1-step20 | 1.750 / 2.000 | 0.917 / 0.833 | 1.417 / 1.417 | 0/4 / 0/4 |
| lr1-step80 | 2.083 / 2.417 | 0.917 / 0.750 | 1.833 / 1.833 | 0/4 / 0/4 |
| lr1-step188 | 2.083 / 2.250 | 0.667 / 0.917 | 1.583 / 1.667 | 0/4 / 0/4 |
| lr3-step20 | 2.167 / 2.083 | 0.667 / 0.583 | 1.500 / 1.083 | 0/4 / 0/4 |
| lr3-step80 | 2.000 / 2.000 | 0.917 / 0.750 | 1.833 / 1.667 | 0/4 / 0/4 |
| lr3-step188 | 2.333 / 2.417 | 0.917 / 0.917 | 1.583 / 1.750 | 0/4 / 0/4 |
| stage18step80 | 2.167 / 2.333 | 0.917 / 0.667 | 1.917 / 1.667 | 0/4 / 0/4 |
| stage18step188 | 2.167 / 2.417 | 0.750 / 0.750 | 1.583 / 1.750 | 0/4 / 0/4 |

### HE

| Model | Grammar 42 / 137 | Scene 42 / 137 | Voice 42 / 137 | Both relations correct 42 / 137 |
| --- | --- | --- | --- | --- |
| selected | 1.312 / 1.250 | 0.688 / 0.438 | 1.375 / 1.375 | 0/4 / 0/4 |
| parent | 1.125 / 1.500 | 0.438 / 0.500 | 1.188 / 1.375 | 0/4 / 0/4 |
| lr1-step20 | 1.438 / 1.500 | 0.562 / 0.438 | 1.250 / 1.312 | 0/4 / 0/4 |
| lr1-step80 | 1.250 / 1.438 | 0.750 / 0.625 | 1.250 / 1.375 | 0/4 / 0/4 |
| lr1-step188 | 1.500 / 1.500 | 0.750 / 0.625 | 1.250 / 1.250 | 0/4 / 0/4 |
| lr3-step20 | 1.312 / 1.312 | 0.625 / 0.375 | 1.438 / 1.312 | 0/4 / 0/4 |
| lr3-step80 | 1.500 / 1.438 | 0.750 / 0.812 | 1.500 / 1.562 | 0/4 / 0/4 |
| lr3-step188 | 1.438 / 1.562 | 0.562 / 0.688 | 1.562 / 1.562 | 0/4 / 0/4 |
| stage18step80 | 1.250 / 1.375 | 0.750 / 0.500 | 1.500 / 1.312 | 0/4 / 0/4 |
| stage18step188 | 1.688 / 1.812 | 0.500 / 0.625 | 1.250 / 1.438 | 0/4 / 0/4 |

### EN

| Model | Grammar 42 / 137 | Scene 42 / 137 | Voice 42 / 137 | Both relations correct 42 / 137 |
| --- | --- | --- | --- | --- |
| selected | 2.500 / 2.125 | 1.250 / 1.250 | 1.875 / 2.250 | 0/1 / 0/1 |
| parent | 2.750 / 2.750 | 1.125 / 1.125 | 2.125 / 1.875 | 0/1 / 0/1 |
| lr1-step20 | 3.000 / 3.000 | 1.375 / 1.250 | 2.625 / 2.000 | 0/1 / 0/1 |
| lr1-step80 | 2.875 / 2.875 | 1.250 / 1.000 | 1.750 / 1.875 | 0/1 / 0/1 |
| lr1-step188 | 2.625 / 2.500 | 1.375 / 1.500 | 2.375 / 2.000 | 0/1 / 0/1 |
| lr3-step20 | 2.875 / 2.625 | 1.500 / 1.375 | 2.375 / 1.375 | 0/1 / 0/1 |
| lr3-step80 | 3.000 / 3.000 | 1.375 / 1.125 | 2.125 / 2.125 | 0/1 / 0/1 |
| lr3-step188 | 2.875 / 2.750 | 1.250 / 1.375 | 2.250 / 2.125 | 0/1 / 0/1 |
| stage18step80 | 2.750 / 2.625 | 1.125 / 1.125 | 1.750 / 1.875 | 0/1 / 0/1 |
| stage18step188 | 2.875 / 2.875 | 1.375 / 1.000 | 2.125 / 2.125 | 0/1 / 0/1 |

## Literal counters on fresh questions

Each language/seed denominator is retained. Word overlap is a flag for reading, not an automatic semantic error.

| Model | Language | Seed | N | подал forms | שנית openings | Answers with mixed-script words | Prompt word overlap (whole answer) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lr1-step188 | he | 137 | 16 | 0 | 0 | 0 | 1 |
| lr1-step188 | he | 42 | 16 | 0 | 0 | 0 | 0 |
| lr1-step188 | ru | 137 | 12 | 1 | 0 | 0 | 1 |
| lr1-step188 | ru | 42 | 12 | 2 | 0 | 0 | 1 |
| lr1-step20 | he | 137 | 16 | 0 | 0 | 0 | 0 |
| lr1-step20 | he | 42 | 16 | 0 | 0 | 0 | 0 |
| lr1-step20 | ru | 137 | 12 | 2 | 0 | 0 | 0 |
| lr1-step20 | ru | 42 | 12 | 5 | 0 | 0 | 2 |
| lr1-step80 | he | 137 | 16 | 0 | 0 | 0 | 0 |
| lr1-step80 | he | 42 | 16 | 0 | 0 | 0 | 1 |
| lr1-step80 | ru | 137 | 12 | 0 | 0 | 0 | 1 |
| lr1-step80 | ru | 42 | 12 | 1 | 0 | 0 | 1 |
| lr3-step188 | he | 137 | 16 | 0 | 0 | 0 | 1 |
| lr3-step188 | he | 42 | 16 | 0 | 0 | 0 | 0 |
| lr3-step188 | ru | 137 | 12 | 0 | 0 | 0 | 1 |
| lr3-step188 | ru | 42 | 12 | 0 | 0 | 0 | 2 |
| lr3-step20 | he | 137 | 16 | 0 | 0 | 0 | 1 |
| lr3-step20 | he | 42 | 16 | 0 | 0 | 0 | 1 |
| lr3-step20 | ru | 137 | 12 | 1 | 0 | 0 | 1 |
| lr3-step20 | ru | 42 | 12 | 3 | 0 | 0 | 2 |
| lr3-step80 | he | 137 | 16 | 0 | 0 | 0 | 0 |
| lr3-step80 | he | 42 | 16 | 0 | 0 | 0 | 2 |
| lr3-step80 | ru | 137 | 12 | 0 | 0 | 0 | 0 |
| lr3-step80 | ru | 42 | 12 | 0 | 0 | 0 | 0 |
| parent | he | 137 | 16 | 0 | 0 | 0 | 0 |
| parent | he | 42 | 16 | 0 | 0 | 0 | 0 |
| parent | ru | 137 | 12 | 1 | 0 | 0 | 0 |
| parent | ru | 42 | 12 | 7 | 0 | 0 | 0 |
| selected | he | 137 | 16 | 0 | 0 | 1 | 0 |
| selected | he | 42 | 16 | 0 | 0 | 0 | 0 |
| selected | ru | 137 | 12 | 0 | 0 | 1 | 1 |
| selected | ru | 42 | 12 | 1 | 0 | 2 | 0 |
| stage18step188 | he | 137 | 16 | 0 | 0 | 0 | 1 |
| stage18step188 | he | 42 | 16 | 0 | 0 | 0 | 0 |
| stage18step188 | ru | 137 | 12 | 1 | 0 | 0 | 1 |
| stage18step188 | ru | 42 | 12 | 1 | 0 | 0 | 1 |
| stage18step80 | he | 137 | 16 | 0 | 0 | 0 | 0 |
| stage18step80 | he | 42 | 16 | 0 | 0 | 0 | 0 |
| stage18step80 | ru | 137 | 12 | 0 | 0 | 0 | 1 |
| stage18step80 | ru | 42 | 12 | 2 | 0 | 0 | 2 |

## Sixteen-turn conversations

| Model | Language | Grammar | Continuity | Voice | Recall turns fully satisfied |
| --- | --- | ---: | ---: | ---: | --- |
| selected | ru | 2.062 | 0.938 | 1.125 | 0/2 |
| selected | he | 1.625 | 0.312 | 0.688 | 0/2 |
| selected | en | 2.875 | 0.688 | 1.812 | 0/2 |
| lr1-step80 | ru | 2.250 | 0.562 | 1.438 | 0/2 |
| lr1-step80 | he | 1.688 | 0.500 | 1.062 | 0/2 |
| lr1-step80 | en | 2.938 | 0.688 | 1.562 | 0/2 |
| lr1-step188 | ru | 2.188 | 1.000 | 1.812 | 0/2 |
| lr1-step188 | he | 1.438 | 0.625 | 0.812 | 0/2 |
| lr1-step188 | en | 2.938 | 0.562 | 1.562 | 1/2 |
| lr3-step80 | ru | 2.125 | 0.750 | 1.750 | 0/2 |
| lr3-step80 | he | 1.562 | 0.750 | 1.000 | 0/2 |
| lr3-step80 | en | 2.938 | 1.062 | 1.938 | 0/2 |
| lr3-step188 | ru | 2.188 | 1.125 | 1.938 | 0/2 |
| lr3-step188 | he | 1.750 | 0.625 | 1.062 | 0/2 |
| lr3-step188 | en | 2.875 | 0.750 | 1.312 | 0/2 |

## Saved BF16 language-body changes

Measured against the same stage-12 parent, comparing exports as FP32 with float64 reductions.

| Arm | Step | Changed elements | Changed fraction | RMS change | Relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lr1 | 20 | 42667242 | 0.085658 | 1.8709073e-06 | 0.00011535825 |
| lr1 | 80 | 110765258 | 0.222370 | 8.242795e-06 | 0.00050824239 |
| lr1 | 188 | 128946664 | 0.258870 | 1.0561629e-05 | 0.00065121938 |
| lr3 | 20 | 105276133 | 0.211350 | 7.329228e-06 | 0.00045191278 |
| lr3 | 80 | 215818566 | 0.433272 | 2.4478964e-05 | 0.0015093481 |
| lr3 | 188 | 237593701 | 0.476987 | 2.9244096e-05 | 0.0018031614 |

## Reading the comparisons

- At final update 188, automatic RU grammar is 2.083/2.250 for `1e-6` and
  2.333/2.417 for `3e-6` on seeds 42/137. HE is 1.500/1.500 versus
  1.438/1.562: the two-seed mean is 1.500 in both arms. EN is
  2.625/2.500 versus 2.875/2.750. Direct reading still finds ordinary malformed
  words missed by the separate grammar pass; those notes are retained alongside
  these automatic ratings and the external blind packet.
- All requested owner/holder facts together remain correct in 0/8 RU and 0/8
  HE answers at each new tested checkpoint (four questions, two seeds).
  Separate extraction preserves correct individual owner claims when the holder
  is missing or contradicted. It does not convert those partial successes into
  full-scene successes.
- All four past-tense `подал` forms occur in 7/12 and 1/12 parent RU answers
  on seeds 42/137; in 2/12 and 1/12 at the `1e-6` final point; and in 0/12 on
  both seeds at `3e-6` updates 80 and 188. These are literal word counts.
- The fresh questions produce zero exact `שנית/שניתן` openings in every
  model/seed group. They did not reproduce the stage-18 opening failure.
  The consumed stage-18 trigger scenes remain a separate diagnostic to replay;
  their old counts are not compared numerically with the new-set zeros.
- The stage-7 primary has mixed Latin/Cyrillic words in 2/12 and 1/12 RU
  answers and mixed Latin/Hebrew text in 1/16 HE answers on seed 137. The new
  tested checkpoints have none in these fresh RU/HE samples. A primary Hebrew
  answer also contains Chinese characters, recorded separately from this
  specific mixed-word counter.
- The long Hebrew conversations repeat whole earlier answers exactly:
  `1e-6 / 188` repeats turn 14 at turn 15; `3e-6 / 80` repeats turn 2 at
  turn 15; `3e-6 / 188` repeats turn 2 at turn 14. Within-answer repeated
  four-gram fractions are zero in these conversations, while the cross-answer
  fraction reaches 1.0. Both measurements are retained.
- In the final `3e-6` English conversation, `without demanding` appears on
  turns 5, 6, 8 and 12. The exact phrase appears in 4/489 English training
  targets. This post-reading phrase check is separate from the frozen counters.
  The same conversation reverses the shoe's behavior under standing/walking
  conditions and confuses the scarf gift. Its automatic voice score is
  1.3125/3 versus 1.8125/3 for the primary on this conversation.
- The short English pocket scene replaces the primary's repeated `His pockets`
  construction with developed tenant scenes in both final arms. One literal
  `1e-6 / 188` excerpt is: “Now the pockets are tenants too; the young man keeps his scarf and the old man keeps his coat.”
  Its exact byte span, source/text hashes and checkpoint path are verified in
  `publication-quote-receipts.json`.
- All nine open-prefix probes at each new checkpoint immediately generate the
  native empty thinking closure and a nonempty final answer. The parent and
  historical stage-18 points do the same. The stage-7 primary closes none of
  the nine opened blocks. These probes retain raw tokens; native EOS and
  thinking closure are measured separately.
- Photographic answers recognize the cat and coffee in English, with inaccurate
  added details. The primary correctly describes the woman in an orange flight
  suit; both new final points call her Dubrovsky instead. Russian cup naming
  improves over the primary's teapot/Chinese-character answer, while ordinary
  agreement and scene details remain defective. All 36 answers were read against
  the inspected photographs. FR/DE/ES each have two fresh prompts per seed;
  these spot checks retain visible reference and idiom errors.

## Historical reproduction and selection

The repeated `1e-6` A40 run ends at validation loss 2.553573, versus 2.553602
for historical stage 18 on L40S. The largest recorded curve difference is
0.000818. Both sets of saved weights were then generated on the same A40 with
the same prompts/settings/seeds. Exact fresh-answer matches are 6/42 and 8/42
at update 80, and 7/42 and 11/42 at update 188. The historical points therefore
stay separate from the matched A40 learning-rate comparison.

The primary stays at stage 7. The external packet has eighteen preselected new
RU/HE questions and 54 literal answers: primary, matched `1e-6 / 80`, and
`3e-6 / 188`, with letters shuffled independently per question. Seed 42 was
fixed before generation. Within the `1e-6` arm, update 80 has the stronger
Russian short-answer voice/grammar combination; within the `3e-6` arm,
update 188 has the strongest automatic Russian grammar average. The packet
awaits Fable's reading. Its key and complete answer sources remain private.

The next data work targets named versus first/second-person ownership, one
then two objects, explicit gifts/loans/safekeeping, Hebrew ordinary inflections,
and maintaining fresh responses through changes of scene. Manual inspection
found that all forty new Russian reference-category training questions name
Лёха explicitly; the new evaluation more often switches “my/your” between
speakers. English character and photographic retention remain measured alongside
language gains. New acceptance scenes are required after this round's diagnosis.

Private weights use `working/lr-control-stage19-{lr1|lr3}-step-XXXXXX` in the
model repository `ataeff/Dubrovsky`. Training/evaluation data, original reviews,
isolated rereads, literal quote receipts and run inventories are backed up under
`additions/2026-11-04-lr-control` in the private dataset repository of the same
name. Final `1e-6` export commit: `809b886d525f6817658dcd671ac3a8bf9049b290`;
weight SHA-256: `8a49bc704c025f9f0ac128844254febe60a5075cf0653b6abbaa0593f8d760cd`.
Final `3e-6` export commit: `04e78daa7114ffbce835552a2db52119c91e48e7`;
weight SHA-256: `45d9d3d8f2762841d87876aa6d63f64fa33327b10909ebf4736d2ea412db750b`.
The owned A40 is stopped (`EXITED`, runtime absent).

Validation: ten text-diagnostic regression tests passed locally;
the original GPU bootstrap suite passed 58 checks before the final two
diagnostic tests were added. The completed-run validator checked all 1,296
generations, token exposure, export hashes and eighteen private weight
inventories. Both published English excerpts passed the literal-quotation
verifier. No new DPO or NoTorch runtime change is included in this experiment.
