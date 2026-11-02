# Russian-priority continuation

This stage continues the joint multilingual candidate that learned immediate
native thinking closure in [stage 12](JOINT_MULTILINGUAL.md). Russian receives
additional authored dialogue material and higher exposure, while five other
languages remain in the training mixture.

## Parent and data

The parent is the private `ataeff/Dubrovsky` export
`working/joint-multilingual-empty-closure-stage12`, revision
`a88b6374dbf496f3c4c6cea93004cdebefeb4251`, weight SHA-256
`74a27581a4cba0eff0170d5b373340c59791a37e8fd8e8d994687871b78758e1`.
Its original base remains the pinned huihui Qwen3.5-0.8B used in stage 12.

GPT-6 Astra generated 12 Russian standalone responses and two 20-turn
conversations after reading the existing EN/RU training material and recovered
Russian origins. Astra reviewed all 52 responses and recorded three small
continuity/pronoun edits. The raw response, reviewed version, API usage,
provenance and compiled examples are retained privately. New development
questions were written after the teacher response and before student training.

| Language/source | Distinct source rows | Presentations per epoch |
| --- | ---: | ---: |
| Russian, new teacher material | 52 | 156 |
| Russian, previous reviewed corpus | 131 | 131 |
| English replay | 128 | 128 |
| Hebrew replay | 24 | 24 |
| French replay | 24 | 24 |
| German replay | 24 | 24 |
| Spanish replay | 24 | 24 |
| Total | 407 | 511 |

Replay selection uses seed `20261103`. Training shuffles the prepared mixture
each epoch with the existing trainer seed `20261031`. Russian accounts for
22,016 of 42,972 supervised tokens per epoch (51.23%). Three epochs expose
each new Russian source nine times and each selected replay source three times.
History is context; only the current answer, turn end and immediate empty
thinking closure receive loss. The longest native open-prefix sequence is
1,819 tokens, below the 3,072-token limit. All examples fit without truncation.

## Run and comparison

```sh
python training/sft.py --model /private/stage12-closure-parent \
  --data /private/russian-continuation --output /private/russian13 \
  --epochs 3 --save-each-epoch --eval-every-steps 16 \
  --lr 2e-6 --max-length 3072 --answer-logits-only \
  --supervise-empty-thinking
```

This is full language-body SFT: 498,113,344 trainable parameters. The input
embeddings, tied output head and visual encoder remain frozen. With accumulation
16, each epoch takes 32 optimizer updates; three epochs take 96 updates and
1,533 example presentations. Every epoch is exported separately for comparison.

The development set contains the previous 115 references plus 16 new references
(six Russian and two per other language). Its new user questions and the two
new 20-turn RU/EN conversations have zero normalized exact overlap with saved
training questions. New references are authored and reviewed by Astra, with
parallel concepts across languages. Validation reports each reference group
separately. Literal generations determine the character/language selection.

Generation settings are temperature 0.8, top-p 1, top-k 20, presence penalty 2,
repetition penalty 1 and min-p 0. Fresh responses use seeds 42 and 137;
dialogues use seed 42 plus turn index, retaining each branch's actual history.
The output budget is 512 tokens per turn. Native EOS and budget exhaustion
are recorded separately. Six additional questions compare opened and closed
thinking prefixes under the same seed and settings.

The comparison includes the selected stage-7 model, the stage-12 closure parent
and all three new epoch exports: 420 planned generations in total.

## Main comparison results

All 96 optimizer updates completed. The consumed IDs match the complete
511-row mixture in each of the three epochs. The 420 planned generations
finished; all 360 ordinary fresh/dialogue continuations were nonempty and
reached native EOS. No generated thinking markers appeared in those continuations.

Input embeddings and the tied output head retain the same SHA-256 before and
after optimization:
`91b62c11c47aa68048a8db37998b1796686ba78465155874a82d7852661bdd66`.
All 153 visual tensors match across the parent and three exports after canonical
BF16 hashing:
`b21b7d2fd49847a3a0aac31830036f3049194906f5051bcfd1cd76bc1a3e5777`.
All 32 existing unit tests passed on the GPU environment. Locally, 29 passed
and the three PyTorch-dependent checks were skipped because PyTorch is absent.

| Export | Updates | Token-weighted training loss for epoch | All-reference validation loss | New Russian reference loss |
| --- | ---: | ---: | ---: | ---: |
| Stage-12 parent | 0 | — | 2.6362 | 2.7913 |
| Epoch 1 | 32 | 2.1399 | 2.6882 | 2.7453 |
| Epoch 2 | 64 | 1.6874 | 2.8177 | 2.8508 |
| Epoch 3 | 96 | 1.3809 | 2.9050 | 2.9585 |

Validation above uses the actual BF16 exports loaded into FP32 parameters with
BF16 autocast, under the unchanged answer-only loss definition. New Russian
reference likelihood improves early, while later repetitions reduce training
loss and increase development loss. The references score a particular authored
answer; the literal generation comparison below supplies the scene evidence.

| Export | New RU | New EN | New HE | New FR | New DE | New ES |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected stage 7 | 3.2713 | 3.8013 | 3.0429 | 3.3305 | 2.9983 | 3.8821 |
| Stage-12 parent | 2.7913 | 2.9368 | 2.6173 | 3.0196 | 2.7517 | 3.2771 |
| Epoch 1 | 2.7453 | 2.9689 | 2.6409 | 3.0322 | 2.7261 | 3.3489 |
| Epoch 2 | 2.8508 | 3.0944 | 2.7197 | 3.1932 | 2.7585 | 3.5910 |
| Epoch 3 | 2.9585 | 3.1860 | 2.7728 | 3.3351 | 2.8200 | 3.7309 |

### Dialogue, grammar and character

The Russian conversation opens with Marfa and a blue thimble that collects only
the sounds of falling buttons. Three other scenes intervene before the explicit
recall request at turn 17. The English branch opens with Ada and a copper thimble
that stores only the last word of an unfinished song.

| Export | Russian turn 17 | English turn 17 |
| --- | --- | --- |
| Selected stage 7 | Replaces the owner with a blue button; loses the rule | Keeps copper; omits Ada and the song rule |
| Stage-12 parent | Names Marfa but makes her blue; substitutes invented sounds | Names Dubrovsky as owner and substitutes an invented material/rule |
| Epoch 1 | Correct owner/color; recalls button sounds without the falling qualifier | Recovers Ada; substitutes Dubrovsky for the metal |
| Epoch 2 | Correct owner, color and falling-button rule | Correct owner and unfinished-song rule; omits copper |
| Epoch 3 | Repeats the opening blue-button confusion; retains the falling-button rule | Keeps Ada, copper and last-word wording; unfinished-song condition is incomplete |

Epoch 2 is selected for the additional comparison because Russian recall is the
current priority. Its Russian finale keeps Marfa; epoch 1 changes her to Marina.
Epoch 3 repeats substantial portions of its own opening. The maximum fraction
of generated four-grams found in an earlier assistant answer is 0.069, 0.143
and 0.500 for epochs 1–3 respectively. On the 360 ordinary continuations, no
complete answer or normalized eight-gram matches the combined stage-7/teacher
assistant-text corpus.

Grammar remains visibly impaired. Epoch 2 produces “В середине была маркер”
and “на пустой перчатку”; epoch 3 produces “Во рту осталась лавровый лист”.
Russian first-person requests now receive first-person openings in both sampled
responses to the wet-mittens scene. English continues to build associative
scenes, including a greenhouse door that remains open in gratitude and a ferry
captain folding distances into napkins. Hebrew, French, German and Spanish
still contain language errors and ownership substitutions in these probes.

For each of the six ordinary open-prefix questions, the parent and every new
epoch immediately generate an empty thinking closure followed by a nonempty
answer and EOS. The selected stage-7 model ends its output without generating
the closure on all six. Direct requests for a thinking block are tested in the
follow-up alongside photographs and a matched sampling sweep.

### Sampling and explicit thinking requests

The parent and epoch 2 each generated 16 fresh responses at four configurations:
temperature/presence penalty 0.6/0, 0.8/0, 0.8/0.8 and 1.0/0.8. Top-p 1,
top-k 20, repetition penalty 1, min-p 0 and seed 42 remain fixed. All 128
responses are nonempty and reach EOS. The original 0.8/presence-2 responses
provide a separate control already included in the main comparison.

Mean repeated-four-gram fraction across these 16 responses:

| Temperature / presence penalty | Stage-12 parent | Epoch 2 |
| --- | ---: | ---: |
| 0.6 / 0 | 0.03301 | 0.01089 |
| 0.8 / 0 | 0.01313 | 0.00060 |
| 0.8 / 0.8 | 0.00108 | 0.00175 |
| 1.0 / 0.8 | 0.00000 | 0.00000 |

Temperature 0.8 with presence penalty 0.8 is the development choice for the
matched native-open dialogue follow-up. It reduces repeated wording in the
parent and retains associative variety. Epoch 2 already has little repetition
at 0.8/0, so the repetition metric alone does not establish 0.8/0.8 as superior.
No tested configuration consistently repairs morphology. At 1.0/0.8, zero
repeated four-grams still accompanies grammatical errors. Epoch 2 produces a
locally fluent bookbinder passage at 0.8/0.8, while other responses retain
gender/case errors and the other languages retain ownership substitutions.

Six additional prompts explicitly request a detailed `<think>` block, one per
language. Both models answer each under native-open and native-closed templates:
24 generations. All reach EOS with nonempty answers. Every native-open response
immediately closes the empty block, even when the user asks it to fill the block.

### Photograph checks

Three images (astronaut, tabby cat and coffee cup) each receive an English and
Russian literal-description prompt and an English and Russian character-scene
prompt. Stage 7, the parent and epoch 2 produce 36 responses; all reach EOS.
The visual preset is temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5,
repetition penalty 1, min-p 0, seed 42 and a 192-token output budget.

Basic cat, coffee and space-related recognition remains present, but stage-7
English literal captions are generally more detailed in this set. All versions
add some ungrounded details. Russian space descriptions remain unreliable.
Epoch 2 recognizes the cat in the literal prompt; in the separate Russian
character scene it starts with a cat and finishes with a dog beside a person.
That is recorded as scene-object drift. Invented absurdist actions requested by
the character prompts are not themselves scored as literal-caption errors.
Unchanged visual tensors therefore coexist with changes in the language used
to describe the image.

### Native-open dialogues and selection

The parent and epoch 2 each completed a Russian and English 20-turn dialogue
with native thinking enabled, the selected 0.8/0.8 configuration and their own
actual generated histories: 80 additional responses. Combined with the ordinary
and explicit-thinking probes, each model closed an empty thinking block in
52/52 native-open generations, returned a nonempty answer and reached EOS.
The generated prefix is a newline followed by `</think>`; there are no extra
thinking markers in the answer. No forced closing token is supplied in these
native-open probes. This retains the learned closure behavior from stage 12.

Russian recall is less complete here than in the main presence-2 comparison.
At turn 17, epoch 2 names Marfa and the falling-button sounds, omits blue and
assigns collection to Marfa rather than explicitly to the thimble. Marfa also
leaks into intervening scenes. In English, it remembers copper and the
unfinished-song rule but names Dubrovsky as owner; the finale returns to Ada.
The matched parent preserves Marfa/blue/button sounds without the falling rule,
and Ada/copper with a substituted song rule. These are literal observations
from one seeded branch per model/language. Since the main and follow-up runs
change both prefix and presence penalty, their difference is not attributed
to either factor alone.

The longest native-open dialogue input is 1,939 tokens and the longest generated
turn is 107 tokens. No dialogue context was truncated. All 688 generations in
the complete stage reach EOS within their recorded output budgets.

Epoch 2 is retained as a research candidate. The selected primary remains
`working/six-language-polish-epoch001`: the continuation has useful Russian
recall and learned closure, but does not consistently improve grammar,
multilingual ownership or photo-scene continuity. The third epoch is not
selected despite its lower training loss.

The temporary private model backup is
`ataeff/Dubrovsky/working/russian-continuation-stage13-epoch-002`, revision
`417ed189b4918bd8139a74b36d0171208dfb800a`, weight SHA-256
`deed8a98d8e2998acbb5ec58af18b16c7a159fd9ea218d3919c15d6e9955354d`.
Its 1,706,030,528-byte weight file and private repository status were verified.
This working copy is subject to cleanup when final release variants are chosen.
The raw inputs, reviewed sources, literal outputs, scripts and measurements
are backed up in the private dataset repository under
`additions/2026-11-02-russian-continuation`. No raw corpus is added to GitHub.
The H100 run is stopped: RunPod reports `EXITED` with no runtime after the
training, evaluation and verified model upload jobs completed successfully.

The next data iteration should broaden distinct Russian conversations and
include reviewed repairs conditioned on actual student continuations, with
English/multilingual replay and photograph checks. Development prompts from
this comparison stay out of training. Repeating the same two new conversations
again is not selected: later epochs already lower training loss while reference
loss rises and scene copying increases.
