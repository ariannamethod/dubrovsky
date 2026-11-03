# Matched grammar comparison, stage 16

Two matched language-body SFT runs produce local improvements, but neither repairs
Russian/Hebrew grammar or reliably preserves scene rules across twenty turns.
The selected primary remains stage 7. Both exports are temporary research
candidates, backed up privately; neither replaces the primary.

Subsequent [stage-17 inspection](REPAIR_TARGET_AUDIT.md) found a defect in the
student-repair objective: preserving the student's events also preserved known
contradictions of user-established scene facts. Both arms contain this defect.
The recorded stage-16 results remain valid observations of those curricula, but
they do not isolate grammar repair from training on those incorrect scene targets.

This executes the data-objective comparison in the
[language method review](LANGUAGE_METHOD_REVIEW.md). Official Qwen, original
huihui and two untuned Gemma instruction checkpoints provide diagnostic
references. They expose inherited language defects on these prompts, without
establishing that every defect is inherited or ranking whole model families.

## Matched training

Both arms start from the same stage-12 immediate-closure parent. Literary uses
broad scene expansion and rewrites; focused uses grammar-targeted scenes and
minimal corrections conditioned on actual student histories. Neither uses DPO.

Each contains 696 common replay records, 79 new RU/HE scenes, 32 student-history
repairs and 79 shared English counterparts. All 886 IDs have the same training
order. Supervised token totals match exactly within each language **between the
arms**; the languages do not receive equal budgets.

| Language | Records per arm | Supervised tokens per arm |
| --- | ---: | ---: |
| English | 307 | 25,652 |
| Russian | 316 | 24,195 |
| Hebrew | 215 | 19,935 |
| French | 16 | 1,618 |
| German | 16 | 1,664 |
| Spanish | 16 | 1,305 |
| Total | 886 | 74,369 |

Counts include the empty thinking closure and assistant turn end. Earlier
history conditions the answer but remains masked as targets. Maximum training
sequence length is 2,356 against a 3,072 limit; no record is truncated. Exact
duplicate full histories and exact train/validation examples are absent. Fresh
question strings were checked against training user turns; this does not prove
complete semantic-family separation.

GPT teacher requests received the authored Russian origins, prior 728-row
mixture and actual training repair histories, without the new holdout prompts.
Every generated field was read before selection. Recorded edits fix Hebrew
agreement/inflection, Russian ambiguity and a Latin character, and shorten
eleven literary repairs to permit exact token matching. The deterministic
budget matcher retains 40 Russian and 39 Hebrew new pairs, omitting eight/nine
respectively without using evaluation outcomes. All six grammar categories
remain represented in each language.

| Setting | Both arms |
| --- | --- |
| Training | Full language-body SFT; 498,113,344 trainable parameters |
| Structure | 852,985,920 total parameters; VLM retained |
| Epochs / updates | One presentation per record; 56 optimizer updates |
| Accumulation / LR | 16 records per update except the final six; 2e-6 |
| Seed | 20261031 |
| Precision | FP32 trainable parameters/optimizer, BF16 autocast/export |
| Libraries | Transformers 5.22.1; torch 2.8.0.dev20250319+cu128 |
| Frozen | Input embeddings, tied output head, 153 visual tensors |
| Thinking target | Native open prefix immediately followed by empty closure |

Live frozen-interface checks and an independent read of both BF16 exports
agree with the parent. Embedding/head SHA-256 is
`3247e63f0f265462e2fba5316dfb2819941ea8ed62ab5b2c4904e4aab5b9d7aa`.
The head remains tied; all 153 visual tensors match in canonical BF16. Tensor
integrity and response quality are evaluated separately.

## Evaluation

`score_grammar_pairs.py` scores reviewed grammatical and incorrect continuations
under the same native closed-thinking context, without asking for a choice label.
It records summed/mean answer-token log probability, lengths, token IDs and
ranking disagreements, excluding EOS. Three unit tests cover length-sensitive
ranking, ties and invalid input.

Text generation uses temperature 0.8, top-p 1, top-k 20, presence penalty 0.8,
repetition penalty 1 and a 512-token maximum with native EOS. This stage holds
sampling fixed; it is not a temperature optimization. Photographs retain the
prior vision preset and 192-token maximum.

| Measurement | Scope |
| --- | --- |
| Fresh generations | 22 prompts × two seeds × eight checkpoints = 352 |
| Grammar likelihood | 42 contrasts × six Qwen checkpoints = 252 scored pairs |
| Branched conversations | Five conversations × 20 turns × four checkpoints = 400 |
| Thinking retention | Nine cases × two prefix modes × four checkpoints = 72 |
| Installation/referents | Six cases × two prefix modes × six Qwen checkpoints = 72 |
| Photograph retention | Three images × four questions × four checkpoints = 48 |

There are 944 generations, separate from the 252 contrast pairs. Conversations
retain each model's actual preceding answers in full: two Russian, two Hebrew
and one English. Thinking and photograph cases are reused retention checks;
fresh prose, dialogues and installation prompts are new. Installation probes
use a neutral language-following system message, including the user's example
about installing the model on a computer, without imposing Dubrovsky's identity.

All 400 dialogue turns and all 48 photograph responses reach EOS. Fresh EOS
counts out of 44 are: selected/parent/literary/focused 44 each, official 27,
huihui 25, Gemma-3 29 and Gemma-4 28. A token-limit stop is not automatically a
repetition loop. Different response lengths also limit comparisons of raw error
counts across these checkpoints.

## Loss and grammatical preferences

Token-weighted online training loss is 2.364 for literary and 2.351 for focused,
averaged over the actual update batches. First/last batch loss would confound
progress with different examples. The small fresh correct-reference set has these
answer-plus-EOS losses:

| Checkpoint | RU | HE | EN | FR | DE | ES |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Parent | 2.602 | 1.887 | 2.875 | 2.731 | 3.275 | 2.547 |
| Literary, update 20 | 2.443 | 1.773 | 2.411 | 2.311 | 2.827 | 2.185 |
| Literary, update 56 | 2.499 | 1.811 | 2.594 | 2.458 | 2.962 | 2.314 |
| Focused, update 20 | 2.460 | 1.744 | 2.439 | 2.333 | 2.851 | 2.197 |
| Focused, update 56 | 2.475 | 1.764 | 2.571 | 2.445 | 2.957 | 2.308 |

Both arms' best sampled loss on this set occurs at update 20. Only update-56
weights were exported. Early validation was recorded, but early weights were
not saved: a limitation and departure from the previous plan to compare early
generations. We cannot claim that update 20 would generate better prose. The
next run must retain those intermediate weights.

| Checkpoint | RU preference, mean / sum, of 12 | HE preference, mean / sum, of 12 |
| --- | ---: | ---: |
| Official Qwen | 11 / 11 | 12 / 12 |
| Original huihui | 11 / 11 | 11 / 12 |
| Selected stage 7 | 11 / 11 | 7 / 7 |
| Stage-12 parent | 11 / 11 | 11 / 9 |
| Literary | 11 / 11 | 9 / 9 |
| Focused | 11 / 11 | 11 / 11 |

All six prefer all six English examples by mean score; stage 7 prefers five by
sum and the others six. All prefer all four FR/DE/ES examples under both scores.
Those sets are near ceiling. Length/segmentation changes some English/Hebrew
rankings. The Russian female-narrator contrast also interacts with the
fixed masculine character system context. These counts do not establish fluent
generation or isolate a language's complete grammar.

## Actual answers and dialogue drift

Official Qwen produces “два предложений”; original huihui produces “три тяжелые
шёпта”. The same prompts expose errors in the tuned checkpoints. Focused produces
“с критикой” where the matched literary answer has “с критики”, but focused also
produces “София поднёс”, “на перекрёстку” and “два тонких шёпотов”. This is local
repair, not a solved Russian language. Hebrew still has agreement, construct-state,
inflection and referent failures; focused's “הוועדה נעשה” is one clear example.

Fresh English remains more readable and can sustain character-specific
association: the focused exam scene turns a typo into a qualification, a typist
and a fine. Historically best English is not established. Scene roles can still
change without justification even when the prose is grammatical.

Twenty-turn conversations expose semantic drift in every checkpoint. Focused
correctly recalls the pin's owner, amber color and “promises nobody has made yet”
rule at turn 7. At turn 15 it reverses the rule into waiting until a promise has
been made. Literary recalls the rule at turn 7 but later puts the pin in
Dubrovsky's pocket after the user explicitly sends its original owner away with
it. Selected stage 7 changes “made” to “kept” at turn 7; the parent contradicts
ownership within its otherwise correct recall.

Russian and Hebrew callbacks show related drift and language defects. The
focused Hebrew thimble scene initially recalls owner, color and echo rule;
later answers again distort roles and grammar. Explicit restatement in the last
turn sometimes elicits copying; that is not independent recall. All designated
recall/final turns were inspected, along with scene transitions and the complete
selected/focused English dialogues. This is qualitative review, not a calibrated
native-speaker panel.

Gemma-4-E2B-it gives cleaner Russian in this small sample, while sometimes asking
for clarification or declining the fictional request. Gemma-3-1b-it also makes
Russian errors; its Hebrew includes nonanswers and an English response. Gemma-4
Hebrew retains clear agreement/construct-state errors. These models were not
trained on the character, use different tokenizers and have different parameter
counts. Equal token caps do not imply equal word budgets. Neither substitutes
for huihui as the authorized training base.

A blinded GPT editorial pass reviewed 96 Russian, 96 Hebrew and 64 English
answers, with shuffled IDs and hidden model labels. Raw judgments, quote checks
and limitations are preserved privately. One quote's capitalization was repaired
without changing its score. Some judgments penalize possible wordplay or
recoverable ellipsis; zero grammar-error scores can reward short nonanswers or
wrong-language output. These scores are not a model leaderboard. The editor
belongs to the teacher's model family, not an independent human evaluation panel.

## Native thinking and photographs

The stage-12 parent and both new arms immediately close the native open thinking
prefix in all 15 cases each (nine retention cases and six installation cases),
produce a nonempty final answer and reach EOS. This tests a learned transition
when the runtime deliberately opens thinking; it does not establish erasure of
a reasoning capability or removal of an architectural module. Stage 7 does not
emit the closure in any of those 15 cases: it generates text and reaches EOS
inside the unclosed thinking region. Its normal use still requires the closed
prefix. Primary selection and learned-closure research status remain separate.

Both original Qwen checkpoints hit the 512-token limit in all six native-open
installation probes. Official Qwen produces no final answer in six; huihui in
five. On the user's greeting/installation example, huihui questions who installed
whom, considers the user an AI, and fails to reach the final-answer boundary.
This reproduces the reported failure mode within the measured budget; it is
not proof of an infinite loop. Closed-prefix answers also confuse referents.
The new arms preserve immediate closure but still sometimes repeat the referent
question or reverse its roles. Closure training does not itself teach pronouns.

Photograph responses still recognize major subjects such as a cat, coffee and
an orange space suit, but visible-detail requests expose invented details and
language errors. Literary invents lettering on the astronaut's patch; focused
invents a brown sock on the cat's forehead in Russian. Invented character scenes
are reviewed separately from these visible-object requests. Frozen visual
weights do not establish unchanged image-description accuracy.

## Decision and next experiment

Keep stage 7 as the selected primary. Focused has useful local effects and a
better Hebrew contrast count than literary, but fails the combined grammar and
continuity goal. Matching a small token budget has not demonstrated sufficient
new construction coverage.

Next, expand varied Russian constructions and actual student-response repairs,
retain task-aligned English replay and Hebrew retention, and save weights at
early validation points. Long training dialogues should preserve an object's
owner and operating rule through topic changes, including corrections after
drift. Establish new holdouts before teacher requests; do not silently recycle
these evaluation scenes as training examples. Use a second training order for
a promising result before promotion. Hebrew then gets a separately measured
focused expansion. FR/DE/ES remain retention checks while these priority goals
are pursued. These losses and generations do not support more epochs of the
current mixture as the default next step.

## Reproduction and private artifacts

| Source | Pinned revision |
| --- | --- |
| Stage-12 parent, `ataeff/Dubrovsky` | `a88b6374dbf496f3c4c6cea93004cdebefeb4251` |
| Selected stage 7, `ataeff/Dubrovsky` | `805bc6e493f284be8b20c262b8fe65070c886c27` |
| `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated` | `4813135658fe51b2e535b8e77927d2f23909ce42` |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |
| `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |

Private dataset `ataeff/Dubrovsky`, prefix
`additions/2026-11-03-matched-grammar/`, stores data, reviewed teacher output,
provenance, literal generations, training order, metrics, integrity checks,
editorial judgments and executed-code snapshots. Raw data and weights stay out
of git. Temporary model backups:

- `working/matched-grammar-stage16-literary` at
  `5ce7e218a1e58a3c21b22a5124cc89b83e391dc9`.
- `working/matched-grammar-stage16-focused` at
  `6cd8371235a67908197babd5df9e1e9e26e7e386`.

Upload inventories and LFS hashes match local files. Gemma-4's initial load
failed because its audio submodule rejects SDPA; the retry uses eager attention.
Earlier Gemma-3 SDPA outputs are preserved. Qwen uses the reference PyTorch
recurrent path. Executed scripts and the initial Gemma comparator are retained
with hashes. All 35 GPU unit tests passed, including three tensor-dependent
checks skipped without local PyTorch. This stage claims no NoTorch/GGUF parity.

The completed-run validator checks all generation IDs, seeds, native histories,
input hashes, matched training order/exposure, frozen-interface/vision records
and private weight receipts. The local suite passes 32 tests with three
PyTorch-only skips. GPU compute is stopped after collection and validation;
the temporary remote Hugging Face credential was removed.
