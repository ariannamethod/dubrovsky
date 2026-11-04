# Russian and Hebrew construction coverage, stage 18

This pass is not promoted. The final export preserves the frozen interfaces and
learned immediate thinking closure, and produces fluent short English scenes.
Russian morphology, Hebrew word forms and ownership/recall remain unresolved.
The retained primary is still stage 7; step 80 and final step 188 are preserved
for external blind comparison.

Fable's blind reading of stage 17 identified specific remaining defects:
Russian `подал` used as a generic action verb, repeated openings, incorrect
owners and holders, female first-person drift, and malformed Hebrew words.
This SFT expands those constructions while retaining English character
material and long conversations with actual previous answers as context.

The stage-17 blind key maps his choices to seven early-step-20 answers, three
restored final answers, one original-control answer, and one question for which
he accepted none. His reading also found a recognizable character in both
languages. The original review and its decoded key are preserved privately.

## Data and editorial corrections

The mixture contains 3,004 distinct records: the 886-row restored replay, 878
previously reviewed but untrained Russian/English expansion rows, 120 new
targeted Russian examples, 1,040 new short Hebrew examples, 48 Hebrew dialogue
turns, and 32 repairs of actual stage-12 Russian continuations. Long teacher
conversations contain sixteen turns with topic changes, delayed recall, loans
and returns, and tests of a fictional object's stated operating condition.
Actual-history repairs retain the student's preceding answers as input and
restore the user's established facts in the current target.

| Language | Records | Supervised target tokens |
| --- | ---: | ---: |
| English | 489 | 36,244 |
| Russian | 1,164 | 89,375 |
| Hebrew | 1,303 | 106,612 |
| French | 16 | 1,618 |
| German | 16 | 1,664 |
| Spanish | 16 | 1,305 |
| Total | 3,004 | 236,818 |

The first stage-18 short-example brief produced a defect of our own: an ethical
character trait became a compulsory gratitude/no-debt/departure formula.
All 120 Russian answers explicitly expressed gratitude. Those 960 original
short drafts are retained for provenance but excluded from training. A second
composition pass removes the imposed recipe and varies answer shape; the
revised Russian batch contains three explicit gratitude expressions. The 200
additional Hebrew examples use the revised brief from the outset.

GPT API composition used `gpt-6.1-sol`; source-anchored editing used
`gpt-5.5-2026-04-23`. Every replacement is tied to an exact substring, and the
request, response, usage, original and final text remain available. Direct
review caught errors in the editor itself: it treated Alexey and Dubrovsky as
different owners in one Russian record and reversed speaker/addressee roles in
two Hebrew records. Those changes were undone. The original camera-card joke
in one exchange was also repaired to follow the correct recipient.

Direct review covered all 190 Hebrew ownership clauses, all 48 new Hebrew
dialogue turns, all 32 actual-history repairs, and stratified short examples.
A late lamp-rule violation in a teacher dialogue was corrected before SFT.
Conversely, an editor changed “half a tone in the oven” to “half an hour” even
though the question mentioned piano tuners. The musical association was
restored. Unrequired narrowing of other associations was likewise reverted.

There are no exact duplicate training examples, no shared training/validation
conversation prefixes, and no fresh evaluation questions in the training
questions or histories. The longest encoded training sequence is 2,356 tokens;
the limit is 4,096, with rejection rather than silent truncation.

## Training and evaluation protocol

The parent is the pinned stage-12 native-empty-closure checkpoint, while stage 7
is the retained primary used as a separate generation baseline. Training is one
pass over the full mixture: 188 optimizer updates, accumulation 16, peak learning
rate `1e-6`, and seed `20261104`. This is language-body SFT with frozen input
embeddings, tied output head and visual encoder. The native empty thinking
closure, answer and turn end are supervised. No new DPO run is part of this pass.

The merged trainer saves every strict validation improvement, including the
initial state, and explicit updates 20, 40, 80, 120 and 160. The final update is
also retained. A Secure Cloud L40S runs both fresh baselines and the new training;
the previously stopped A40 had no available GPU to resume.

Before teacher requests, 42 new six-language prompts and three sixteen-turn
RU/HE/EN conversations were frozen separately. Each model's long conversation
contains its own literal answers. Text sampling uses temperature 0.8, top-p 1,
top-k 20, presence penalty 0.8, repetition penalty 1, seed 42 and a 512-token
limit with native EOS. Existing native-thinking and photographic probes are
retained as regression checks. Baselines each contain 42 fresh answers, 48
dialogue turns, 18 closed/open-prefix probes and 12 image-conditioned answers.

The training JSONL SHA-256 is
`e5cdf07ae8367343b73807431f9f1aff3ed2219af2b50d7fb81d6ed938bc82ec`;
validation JSONL SHA-256 is
`ba6c143723b7e08dcee80681525518957248692e14eadb9fdf71723249065c8f`.
Parent model revision:
`a88b6374dbf496f3c4c6cea93004cdebefeb4251`, subdirectory
`working/joint-multilingual-empty-closure-stage12`, weight SHA-256
`74a27581a4cba0eff0170d5b373340c59791a37e8fd8e8d994687871b78758e1`.

## Quotation provenance

`quote_receipts.py` verifies all ten previously supplied Fable quotations against
the saved generation files. Each receipt includes an exact UTF-8 span, source
and text hashes, checkpoint attribution and actual generation settings.
Seven regression tests reject paraphrases, changed sources, ambiguous matches,
incorrect checkpoints/seeds and misattributed sampling. The source datasets and
receipts remain private; the verification utility is public.

## Completed run

The SFT completed all 188 updates in 1,375.24 seconds. Every one of the 3,004
records appeared exactly once; the executed target-token count is 236,818.
Online token-weighted training loss is 2.707527. Ten exports remain available
(including the unchanged parent at step 0); all nine nonzero exports were
verified in private Hugging Face storage. Frozen embedding/head integrity and
all 153 visual tensors match the parent across every export. The tokenizer
also matches. The captured executed-code and input inventories match the local
sources and data hashes.

| Update | Total validation loss | Fresh RU | Fresh HE | Fresh EN |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2.632212 | 2.713391 | 2.370279 | 3.226525 |
| 20 | 2.591004 | 2.643544 | 2.304393 | 3.120046 |
| 40 | 2.579634 | 2.596282 | 2.260576 | 3.071025 |
| 60 | 2.572264 | 2.571788 | 2.213848 | 3.071601 |
| 80 | 2.565071 | 2.553736 | 2.191466 | 3.059411 |
| 100 | 2.557072 | 2.543452 | 2.171966 | 3.045113 |
| 120 | 2.554540 | 2.535877 | 2.155095 | 3.042113 |
| 140 | 2.554758 | 2.530115 | 2.151196 | 3.030467 |
| 160 | 2.555814 | 2.534909 | 2.153233 | 3.036919 |
| 180 | 2.554219 | 2.532643 | 2.151279 | 3.034654 |
| 188 | 2.553602 | 2.527899 | 2.148519 | 3.027696 |

The recorded minimum is update 188. To check the small plateau difference,
saved updates 120 and 188 were rescored in BF16 and FP32 with TF32 disabled,
using the same 16,697 supervised validation tokens. The FP32 run upcasts the
saved BF16 exports; the live pre-export FP32 training tensors are unavailable.

| Saved update | BF16 forward loss | FP32 forward loss |
| ---: | ---: | ---: |
| 120 | 2.554699 | 2.554349 |
| 188 | 2.553792 | 2.552986 |

Update 188 remains lower than 120 in both forward precisions.

## Generation comparison

The completed battery contains 588 generations: 252 fresh six-language answers,
108 open/closed thinking probes, 192 turns in sixteen-turn conversations, and
36 photographic answers. All dialogue histories were checked against the same
model’s literal previous answers and the frozen user turns. Every fresh answer,
dialogue turn and image answer reached native EOS. The stage-7 primary answered
the open thinking probes without closing the thinking block; the stage-12
parent and every new tested export generated immediate empty closures in 9/9
open-prefix probes, including the explicit requests for thinking.

An anonymous GPT-5.5 review scored grammar and character from 0–3, and scene
fidelity/continuity from 0–2. Its raw requests, responses and blind keys remain
private. Literal-evidence validation rejected 18 fresh-answer reviews and four
long-dialogue reviews: most quoted the question; one cited a word from another
answer. Each rejected item was reread in a separate request with its own
context. All final error quotations occur in the corresponding answer. Direct
reading was also used for the final RU/HE/EN answers, complete final dialogues,
baseline comparisons and the photographic probes. The automated scores below
are those of the same model family used for editing the corpus.

Fresh-question mean grammar scores (12 RU, 16 HE and eight EN answers per model):

| Model | RU | HE | EN |
| --- | ---: | ---: | ---: |
| Stage-12 parent | 2.250 | 1.188 | 2.750 |
| Stage-7 primary | 1.500 | 1.125 | 2.750 |
| Update 20 | 1.750 | 1.062 | 2.500 |
| Update 40 | 1.833 | 1.000 | 2.750 |
| Update 80 | 2.083 | 1.125 | 2.750 |
| Update 188 | 1.667 | 1.312 | 3.000 |

Mean continuity scores over each sixteen-turn conversation:

| Model | RU | HE | EN |
| --- | ---: | ---: | ---: |
| Stage-12 parent | 0.500 | 0.250 | 1.000 |
| Stage-7 primary | 0.438 | 0.312 | 0.688 |
| Update 20 | 0.375 | 0.188 | 0.750 |
| Update 188 | 0.500 | 0.500 | 0.875 |

The main observed changes and remaining failures are:

- Russian openings beginning with `Дубровский подал` fell from 8/12 at the
  parent to 2/12 at update 80 and 0/12 at update 188. The final point begins
  3/12 answers with `Дубровский попросил`; broader verb use still needs work.
- The final RU/HE answers correctly supply all requested ownership and current
  holder facts in 0/6 targeted questions. One Hebrew gift question names the
  correct owner, but its holder remains wrong. Partial facts are retained in
  the source answers rather than counted as complete success.
- Russian first-person female forms appear in two final scenes, while the
  Varvara scene mixes masculine and feminine self-reference. The final long
  Russian conversation repeatedly says `янтарное свисток` and reverses the
  return of the whistle. Topic changes continue to carry objects from earlier
  scenes into requests to focus on a new scene.
- Update 40 produced a 386-token Russian Tuesday repetition, ending at native
  EOS with repeated-four-gram fraction 0.491. The same fresh case at updates
  80 and 188 does not repeat that way. Their maximum repeated-four-gram fraction
  over the 42 fresh answers is zero.
- Hebrew has small gains in the final automated grammar/continuity averages,
  while malformed verbs, agreement and pronoun/reference errors remain plainly
  present. In the long scene, lending the bell is incorrectly said to change
  its rule, and an already-told dream is awakened.
- Short English retains coherent associative scenes, including soup becoming
  a map and a window negotiating rest. The long English scene still reverses
  the whistle rule and the cup/scarf ownership. Its final mean grammar score
  is 2.812/3 versus 3.000/3 for the parent on this conversation.
- The photographic check still identifies the main subjects in English
  (astronaut, cat, espresso), with incorrect details such as the helmet’s
  location. Russian adds or confuses objects, including a nonexistent white
  teapot. The frozen visual weights and these observed outputs are recorded
  separately. French, German and Spanish retain visible grammar/reference
  defects in the two fresh probes per language.

## Selection and next controlled run

The primary remains stage 7. The new corpus pass is retained as an experiment;
update 80 is the stronger new Russian short-answer comparison point, and update
188 is the final/reference-loss point. A fresh external blind packet contains
14 RU/HE questions and 42 literal answers from stage 7, update 80 and update
188, with letters reshuffled for each question. Its key and source hashes stay
private. Fable’s prior review covers stage 17; the new packet awaits external
reading.

The next planned control changes peak learning rate from `1e-6` to `3e-6` with
the same pinned stage-12 parent, reviewed corpus, one-pass sample order, seed,
freeze policy and automatic-best exports. Stage 19 completes this comparison; see [LR_CONTROL.md](LR_CONTROL.md).
It will measure the effect of update strength before another corpus expansion.
The stage-18 questions have now been used for selection and diagnosis; new
scenes are required for the next independent acceptance check.

Private weight prefix: `working/ru-he-coverage-stage18-step-XXXXXX` in
`ataeff/Dubrovsky`; final backup commit
`44beda6cfc5ecf72d488dee4787ca05debcd6d5f`. Final-update weight SHA-256:
`79c23e5c56ed3326f41a3123d3cc2cda36ba779658185a439c58a9849610615e`.
Corrected corpora, rejected drafts, source edits, raw generations, evaluator
rechecks and quote receipts are backed up in the private dataset repository.
The owned L40S is stopped (`EXITED`, runtime absent).

Validation: 43 existing unit tests passed on the GPU environment; seven new
quotation-receipt regression tests passed locally; all ten real Fable quotes
passed literal attribution checks; the completed-run validator verified the
588 generations, exact sample exposure, frozen tensors and private weight
inventories. No new DPO or NoTorch runtime change is included in this pass.
