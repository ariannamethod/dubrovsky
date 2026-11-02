# Three-stage lineage audit — 2026-11-02

The selected stage-7 Dubrovsky remains the primary. This evaluation compares the
original huihui, the first English-only SFT export, and stage 7 on identical
inputs. No optimizer ran. It also fixes a duplicated tied-weight export in SFT.

## Models and controlled conditions

- Original: `huihui-ai/Huihui-Qwen3.5-0.8B-abliterated`, revision
  `4813135658fe51b2e535b8e77927d2f23909ce42`.
- Early English: private `ataeff/Dubrovsky/working/english-reconstruction-epoch003`.
- Selected: private `ataeff/Dubrovsky/working/six-language-polish-epoch001`.
  Both private snapshots use revision `805bc6e493f284be8b20c262b8fe65070c886c27`.

All three ran on the same H100 with PyTorch `2.8.0.dev20250319+cu128`,
Transformers `5.22.1`, BF16 inference and the reference recurrent kernels.
Native closed-thinking prompt token IDs matched across the snapshots. The system
message was `You are Alexey Dubrovsky.` The comparison comprised 528 generations:

- 20 fresh scene questions, seeds 42 and 137: 40 answers per model.
- Two RU/EN conversations with 20 user turns each, seed 42, branching on each
  model's actual answers: 40 answers per model.
- 48 new word-choice questions across six languages, seed 42, presence penalties
  0 and 2: 96 answers per model.

Fresh scenes and conversations reuse the held-out stage-10 probes. They remain
outside training, but are development comparisons that have already been viewed.
The 48 new grammar choices and their references are Astra-authored and
self-reviewed. All 108 evaluation user prompts have zero normalized exact overlap
with 1,454 distinct user questions in the saved training mixtures and histories.
This overlap check covers exact normalized questions, not semantic similarity.

Temperature is 0.8, top-p 1.0, top-k 20, repetition penalty 1.0 and min-p 0.
Presence penalty applies once to each distinct generated token ID, excluding
prompt tokens. Scene/conversation outputs use presence 2 and a 192-token limit;
word-choice outputs use a 32-token limit. All use native `enable_thinking=False`
and stop IDs 248044/248046. Full raw text, token IDs and actual histories are saved.

## What English-only SFT changed, and later training recovered

A manual review of all 120 fresh responses records the predominant output
language. A short foreign insertion does not change that label.

| Prompt language | Original huihui | Early English SFT | Selected stage 7 |
| --- | ---: | ---: | ---: |
| English | 12/12 | 12/12 | 12/12 |
| Russian | 12/12 | 3/12 | 12/12 |
| Hebrew | 4/4 | 0/4 | 4/4 |
| French | 4/4 | 3/4 | 4/4 |
| German | 4/4 | 1/4 | 4/4 |
| Spanish | 4/4 | 3/4 | 4/4 |

The early English checkpoint often answers other languages in English despite
frozen embeddings and the tied head. Stage 7 restores the predominant response
language on this sample. Language choice and grammatical quality are separate:
stage 7 still produces Russian agreement errors such as “от своих собственной
истории”, German “in einer dritten Spiegel”, and Hebrew “היא השנייה חתם”.
English also has defects, including “it sigh loudly and complain”.

The base already fails elementary choices: Russian “пять яблоки”, for example.
Its exact correct choices per language are EN 6/8, RU 5/8, HE 4/8, FR 6/8,
DE 5/8 and ES 7/8 under both presence settings. The trained characters frequently
turn these questions into explanations or scenes. Early English begins one German
answer, “Dubrovsky tested positive for German”. Exact-choice scores therefore
combine grammatical choice with format compliance; they cannot be read as a
percentage of language knowledge retained. The scorer separates exact answers,
mentions of either option, mentions of both, and answers containing neither.

Among 32 English scene/dialogue responses per model, early English uses `his`
78 times in 2,380 words and `forever` 33 times. Stage 7 uses `his` 10 times in
1,395 words and `forever` zero times. These describe this sample's wording and
length; character quality still requires reading the scenes.

## Long conversations and termination

At the English opening-scene callback, early English renames Mara to Mica.
Stage 7 assigns the repair to Ivo, a gardener from a later scene, and changes the
bell's trigger. The base also carries Ivo into the requested return to the repair
shop. In Russian, stage 7 recalls Mira, the letter Щ and the blue button at turn
17, but the answer and final turn retain language errors.

All 80 scene/dialogue answers from each trained checkpoint reach EOS and are
nonempty. The base reaches EOS on 9/40 fresh answers and 8/40 dialogue turns;
its other responses meet the shared 192-token cap, often during verbose prose.
These capped outputs are not counted as reasoning loops. No generated thinking
markers occur in the 528 records. The selected stage-7 conversation has zero
prior-answer word-4-gram overlap on this run, while its callback still fails:
absence of repeated phrases does not establish scene memory.

## SFT export defect

The old SFT exporter converted each state-dictionary entry independently. Input
embeddings and the tied output head became two separate CPU allocations, so the
serializer wrote the same 508,559,360-byte tensor twice. Their saved hashes match.
This is a storage defect; the compared model parameters retain the same values.

`export_bf16_state` now preserves exact aliases while keeping distinct views and
independent equal-valued tensors separate. The live FP32 training parameters and
optimizer state remain untouched. DPO already exports through a model-wide dtype
conversion and does not use this defective path.

An actual Qwen round trip compares all 474 reloaded state tensors byte-for-byte,
checks the live FP32 model remained unchanged, and verifies the reloaded head is
tied. The saved weight file shrinks from 2,214,590,296 to 1,706,030,528 bytes,
saving 508,559,768 bytes including metadata. The compact file is an equivalent
serialization on the evaluation pod, not a newly selected or published model. All 40 fresh generations from the
compact export match the original stage-7 token sequences exactly. All 32 unit
tests pass in the PyTorch environment; the local environment passes 29 and
explicitly skips the three PyTorch-dependent export tests.

## Next measured tuning

Keep stage 7 as the parent. Language strengthening should preserve its restored
language choice and character while repairing agreement and scene ownership.
The next preference trial should use minimal language edits where possible;
broad literary rewrites mix grammar repair with a change of voice. Existing
contextual preferences remain prepared, not trained. Evaluate English retention,
fresh multilingual prose and independent topic-switch/callback conversations
alongside any reference loss. Retire any probe used to construct new training
pairs before selection on replacement scenes.

Private inputs, reference answers, full responses, file hashes, manual language
labels, operators and measurements are backed up under
`ataeff/Dubrovsky` (dataset), `additions/2026-11-02-lineage-audit/`.

## Subsequent controlled reconstruction

At Oleg's request, the [joint multilingual comparison](JOINT_MULTILINGUAL.md)
now starts three matched runs from original huihui rather than continuing stage 7.
It tests the historical ordering hypothesis and learned native thinking closure
separately. This subsequent experiment does not change the lineage observations
above or claim to replay the entire earlier training chain.
