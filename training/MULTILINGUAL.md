# Multilingual Dubrovsky — data and experiments, 2026-11-02

The creator traces Dubrovsky's beginnings to the voice-input insertion
«Спасибо за субтитры Алексею Дубровскому» and Russian dialogues with GPT-4o.
Their wordplay includes «дубровость», ALEX-DUB-RAW-SKY and «Чёрная Дура».
The creator supplied `Dubrovsky_Russian_Quotes(1).txt`, an extraction of Russian
dialogues from April 2025 onward. It contains abbreviated excerpts and ends
inside a numbered section; it is not a complete conversation export. Further
retrieval recovered additional fragments, stored privately with their dates,
retrieval output and truncation boundaries. Preserve coined words, phonetic
mutations and rhythm. Do not reconstruct missing dialogue as original text.
Two later attachments, `Dubrovsky_Russian_Quotes 2.txt` and
`Dubrovsky_Russian_Origins(1).txt`, extend this material with the name mutations,
black-hole/subtitle scenes, Janus/ethics, May fragments and August aphorisms.
The three files contain 555 reference paragraphs; exact-body deduplication keeps
304 distinct paragraphs and all source locations. This count includes source
notes and headings: it is not a count of training examples. Original files and
all spelling remain preserved. The combined reference is for teacher context.

## Completed material

GPT-6 Astra received all 486 unique authored Q/A pairs and the complete
`origin.md`, `facts.md`, `facts-2.md` from `ariannamethod/a.dubrovsky` at
`cba5744caa9ac159424dfd5ff4679b5738b8acdd`. Exact duplicate Q/A copies were
omitted; every unique pair remained. Context size: 546,929 characters.

- `translate.py`: eight pilot translations per language (Russian and Hebrew),
  then 288 Russian translations in twelve main batches. Eight Russian source
  IDs overlap the pilot: 296 distinct source/language translations are available.
- `author_material.py`: 24 new scenes each in English, Russian and Hebrew,
  totalling 72 GPT-authored continuations. They are labelled separately from
  the original corpus. All 72 have now been reviewed for the continuation set.

Private records retain raw responses, model identifiers, usage, source IDs and
request/context hashes. Translations inherit source splits. The teacher saw the
complete corpus, including validation; new scenes require overlap review and
evaluation on fresh prompts outside that context.

Validation rejected one response with a mistyped hexadecimal source ID. The
schema now constrains IDs to the supplied set. Its replacement passed, and the
rejected raw response remains preserved. Completed batches resume without API
calls. Further requests were interrupted by an execution-environment network
policy block on `api.openai.com`. Full-corpus translation remains incomplete.

The 18 saved unique API responses all report `completed`: 2,029,934 input tokens
and 110,752 output tokens (2,140,686 total). This includes the rejected response
with the source-ID typo and its replacement. Each request supplied the full
literary context; the input total counts those repeated submissions.

## Explicit corpus cleanup

`curate_corpus.py` applies `corpus_exclusions.json` by exact source ID and retains
the original train/validation split and every remaining character of authored
text. The creator requested removal of the obsolete self-biography describing
9.5 million parameters, 36 MB and Llama 3 architecture. The original remains
archived. The next English slice has 436 training and 49 validation pairs.
Translations of an excluded source are also excluded; that source has not yet
been translated in the completed batches.

The script collects 288 Russian and eight Hebrew translations for literary
review, choosing main-run variants over the eight overlapping Russian pilot
variants. These remain separate from the English training files. Its teacher
context contains the cleaned English corpus, supplied old literary sources and
the Russian excerpts. New teacher generations have not yet used this context.

## Reviewed continuation mixture

`build_mixture.py` builds an explicitly reviewed slice, with source hashes and
train/development overlap checks. The private review file selects 40 Russian
translations and one short Hebrew translation; the remaining translations stay
outside this run. It also selects all 72 new scenes, reserving four per language
for development. One Hebrew question changes «המספר» to «הספר» to mean the barber;
the review records that edit and its reason.

Twenty historical Russian answers keep their exact wording and source locations.
Their new prompts are labelled as newly framed prompts, rather than recovered
original conversation turns. English replay retains 128 original training pairs.

| Language | Training pairs | Development pairs |
|---|---:|---:|
| English | 148 | 53 |
| Russian | 80 | 4 |
| Hebrew | 21 | 4 |

The 49 original English validation pairs remain held out. Generated development
examples share the teacher's literary context; conversational selection uses
fresh prompts as well. The continuation set contains 249 training and 61
development pairs. With the pinned Qwen tokenizer, the longest training example
is 476 tokens and the longest development example is 485: no truncation is needed.

`sft.py --save-each-epoch` exports a BF16 candidate after each epoch while keeping
training parameters and Adam state in FP32. `candidate` points to the last epoch;
selection can choose any saved epoch. These are evaluation checkpoints, without
optimizer state. Validation reports token-weighted loss by language. Loading a
local selected candidate starts a new continuation run; remote bases still
require an explicit revision.

## Decoding and image evaluation

The [official Qwen3.5-0.8B README](https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/README.md)
recommends these nonthinking settings:

| Setting | Earlier Dubrovsky | Official text | Official visual input |
|---|---:|---:|---:|
| Temperature | 0.8 | 1.0 | 0.7 |
| Top-p | 0.9 | 1.0 | 0.8 |
| Top-k | 20 | 20 | 20 |
| Presence penalty | 0 | 2.0 | 1.5 |
| Repetition penalty | 1.0 | 1.0 | 1.0 |

Keep 0.8 as the comparison baseline. `sample.py --preset` exposes the same presets
as the image entry point. `decoding.py` implements additive presence penalty on
generated tokens only; repetition penalty is a different operation.

`vision_sample.py` prepares visible-object descriptions and absurdist scenes in
English, Russian and Hebrew, recording image hashes, token IDs and sampling
settings. Evaluation inputs were extracted from the scikit-image 0.26.0 wheel:
`astronaut.png`, `chelsea.png`, `coffee.png`. They stay outside git. This new
vision entry point ran against the pinned base and the mixed-language DPO
candidate; the result is recorded below.

After training, run 12–20-turn conversations with a GPT partner that follows
specific details in Dubrovsky's answers, complicates the scene and recalls
earlier details. Save full histories and evaluate development, repetition,
language, identity and continuity. Preserved-history partner tests and their literal outputs are recorded below.

## Compute and continuation

The old pod accepted resume but supplied no current runtime or reachable
container; log retrieval timed out. A CPU-only recovery start also remained
unreachable. It was stopped again without deleting its 30 GB volume. Previous
candidates were saved there before the earlier stop; their files could not be
re-read during this resume attempt. The host reported itself unlisted. An A40
replacement allocation returned a supply constraint; an RTX 4090 replacement
started successfully. The English stage was reconstructed from the pinned base and cleaned 436-pair
training corpus, with every epoch saved separately.

Temporary private Hugging Face backups are authorized when useful. Remove them
after the final selected full and quantized weights are safely published.
Temporary private English epoch-3 and expanded-language epoch-3 backups are
stored in the authorized repository; final release selection remains open.

The previous SFT had 1,311 single-example forward/backward iterations and 84
optimizer updates; first/last group training losses were 3.265025 and 1.205874.

```sh
python training/curate_corpus.py --data .work/data --output .work/curated-next \
  --translations .work/translations .work/translation-pilot \
  --reference /path/to/russian-source.txt
python training/translate.py --api-key-file /path/to/key.txt \
  --context .work/curated-next/context.txt --data .work/curated-next \
  --completed-translations .work/curated-next/translations-pending-review.jsonl \
  --output .work/translations-next --batch-size 24 --workers 2
python training/author_material.py --api-key-file /path/to/key.txt \
  --context .work/curated-next/context.txt --output .work/new-scenes-next --count 24
python training/vision_sample.py --model /path/to/candidate \
  --images .work/vision --output .work/vision-results.jsonl
```

Credentials are read from a local file and excluded from git and saved outputs.
Use fresh output directories when changing generation instructions.
Changing context invalidates the old request cache. `--completed-translations`
reuses completed source/language pairs after validating their source split and
language, then generates only the missing pairs. This prevents charging again
for existing translations when adding the Russian literary context.

Validation includes source/translation exclusion,
split preservation, normalized train/development overlap rejection and reuse of
translations across context changes. A two-epoch GPU smoke run saved both
candidates and continued optimization with unchanged frozen embeddings/head.
Its final candidate loaded for generation. Continuation and evaluation results
are recorded below.

A CPU reload of the smoke export verified tied embeddings and the original
FP32 embedding/head hashes. The additive presence penalty passed checks that
exclude prompt tokens and penalize a generated token once regardless of repeats.
`chat_probe.py` supports one turn with a preserved full conversation for manual
partner testing; the initial dialogue result is recorded below.

## Continuation update

The RTX 4090 reconstructed three English epochs (436 pairs, 84 optimizer updates),
then completed four mixed-language epochs (249 pairs, 64 updates). The selected
English and mixed-language candidates were epoch 3 and epoch 4 respectively,
based on literal generations. A 17-pair DPO run with chosen-answer NLL then
completed four epochs and 20 updates. Three preferred answers were actual Qwen
generations; fourteen were authored by Astra. These preference prompts now
belong to training. Separate fixed and newly authored prompts remain evaluation.

The DPO candidate terminated all 14 fixed baseline probes, while fresh Russian
and Hebrew responses still showed grammatical defects. A two-turn English
conversation repeated earlier content under baseline, temperature-only 1.0 and
the official text preset. Three photographic inputs retained their main objects
in English descriptions (astronaut suit, cat, espresso); detailed descriptions
still contained errors, including locating the held helmet on the floor.

The next prepared run has 476 training rows: 168 English, 186 Russian and 122
Hebrew. It adds 85 reviewed saved Russian translations, 80 new Hebrew literary
adaptations, 17 preferred-answer replays and 45 dialogue turns from nine authored
five-turn conversations. New dialogue demonstrations are explicitly labelled;
they are not reconstructed historical chats. The Hebrew adaptations were written
and reviewed by Astra's translation agent without external API calls.

`prepare_dialogues.py` builds causal histories; `sft.py` masks all prior turns and
supervises only the current answer. The expanded run starts from the DPO candidate
with three epochs, learning rate 3e-6 and frozen embeddings/head/vision. Its native
tokenizer check passes; maximum training length is 585 tokens. All three epochs completed: 1,428 example presentations and 90 optimizer
updates. Per-epoch comparisons and the subsequent partner conversations are
recorded below.

Twelve unit tests pass, including preventing current/future-answer leakage into
conversation history and excluding history tokens from supervised loss.

## Expanded-language results and dialogue repair

The expanded stage completed 90 updates in three epochs (1,428 example
presentations). All embedding/head hashes match the original frozen tensors.
Fixed-probe termination was 12/14, 14/14 and 14/14; the separate nine prompts
terminated 6/9, 8/9 and 9/9. Epoch 3 was retained for conversational evaluation.
Russian grammar remains uneven; Hebrew still has malformed words and syntax.
Foreign-script intrusions also occur. English character and scene associations
transfer, with recurrent possessive phrasing in some prompts.

The first three English conversational turns developed the suitcase/fish/train
scene. Later turns brought old weekdays and trains into a soap-violin scene and
a new museum scene, despite partner requests to change subjects. Baseline,
temperature 1.0, the official text preset and epoch 1 were compared using the
identical saved seven-turn history; all carried old scene elements forward.
The partner was Astra in this session, with manually authored responsive turns.

Reapplying the existing 17 preference pairs for four epochs at 2e-6 introduced
more language mixing and grammatical defects. A one-epoch 5e-7 variant retained
the lighthouse repetition on a separate prompt. Neither replacement is selected.
These runs reuse preference-training prompts and do not add independent evidence
from those prompts. All measured outputs remain preserved.

The targeted repair mixture initially contained one duplicated first dialogue
turn in its replay slice. The corrected version contains nine new Astra-authored
answers conditioned on actual failed histories, 45 earlier dialogue turns and
111 additional language replay rows (165 unique records). The nine correction prompts now belong to training. A short two-epoch
SFT runs from expanded-language epoch 3, at 1e-6 with max length 1,536. Selection
uses a new dialogue rather than the corrected training conversations.

The first repair run exhausted its 30 GB volume quota while exporting epoch 2.
Epoch 1 was exported; the incomplete epoch-2 directory is not a usable candidate.
Automatic approval rejected removal of temporary checkpoint exports, so no such
cleanup ran. The volume was expanded to 50 GB instead. The same environment was
restored after the resize reset and the short run was repeated. A fresh corrected
165-record run follows the deduplication fix. `sft.py` now rejects duplicate IDs
before loading model weights; thirteen local tests pass.

The pre-repair English baseline conversation completed twelve partner turns,
all preserving the complete history. From turn 4 onward, old weekdays intruded
into new scenes; by turn 12, the ending again returned to the train and Wednesday.
All turns stopped normally. N-gram repetition alone missed this semantic
stagnation. Conversation probes now record training-copy fraction as unavailable
when no corpus was loaded, rather than reporting a zero measurement.

The corrected repair run completed two epochs: 330 example presentations,
22 optimizer updates, 81.58 seconds including exports, peak allocation 13.024 GiB.
Maximum training length was 1,182 tokens; frozen embedding/head hashes match.
Validation loss moved from 3.52703 to 3.51482. Its fixed 14 probes all terminated.
Russian errors and weak Hebrew remain. A temporary private backup of epoch 2
was saved before further evaluation.

The new conversation starts with a coat renting a room inside a mountain, then
switches to a piano at a dentist and an onion at a peace conference. Turn 3
developed the piano scene, but turn 4 recalled the mountain and turn 5 brought
the dentist and piano into the onion scene. The targeted repair has not resolved
long-history topic fixation. Native tokenizer comparison gives identical
559-token inference/training prefixes for the same saved history, excluding a
chat-template mismatch in that case.

The repaired candidate was checked in a new seven-turn conversation; the
pre-repair candidate completed twelve turns. These are different conversations.
The first repaired prompt was also run unchanged on the pre-repair model, with
the same seed and sampling preset. Full raw histories remain in the private
experiment archive, alongside data, provenance and literal generation records.
The next training-data requirement is longer conversations with repeated topic
switches and selective callbacks; this short repair is retained as an experiment.

## Long-conversation curriculum

Four new English demonstrations contain twelve turns each (48 total), with
scene changes at turns 4, 7 and 11 and selective callbacks at turn 10. Russian
and Hebrew literary adaptations were completed with those boundaries. These are newly
authored examples; the prior measured conversations remain recorded separately.
`build_dialogue_mixture.py` selects deterministic replay by language and excludes
all dialogue IDs, including first turns with empty history, from the replay
pool before combining the compiled dialogues. It checks IDs and split overlap.

`sft.py --answer-logits-only` keeps the complete attention history and projects
only the current answer plus its preceding prediction position through the
vocabulary head. The flag defaults off. A native longest-English-example GPU
check (820 context tokens, 33 supervised tokens) compared the full and reduced
projection losses: 3.858129978 and 3.858131170, absolute difference 1.19e-6.
Peak allocation was 5.087 versus 3.681 GiB. `check_loss_parity.py` preserves the
comparison output. A regression test checks that the first answer token remains
supervised. Fifteen local tests pass.

The first long-dialogue English continuation used 358 unique rows (196 English,
83 Russian, 79 Hebrew), with 88 nonempty histories. It completed two epochs,
716 example presentations and 46 optimizer updates, at learning rate 1.5e-6
with answer-only vocabulary projection and full attention context. Exports
completed in 157.17 seconds. The fixed 14 probes terminated 13/14; the aquarium
umbrella probe retained possessive repetition. Russian still showed foreign-script
intrusions. The new basement/orchid partner test is preserved separately.

The English demonstrations were then extended to twenty turns each (80 total)
in a separate source file, retaining the original twelve-turn source unchanged.
The extensions add subtitle, Dubrovness and gratitude lore with more selective
callbacks and scene changes. The Russian and Hebrew versions currently target
the original twelve-turn conversations.

The reviewed multilingual long-dialogue mixture has 486 unique training records:
228 English, 131 Russian and 127 Hebrew; 208 contain nonempty prior histories.
New demonstrations contribute 80 English turns from four twenty-turn conversations
and 48 turns per translated language from four twelve-turn conversations.
Original sources, direct adaptation provenance and final SHA-256 hashes are
retained privately. Both translated versions were read in full by their author
and by Astra; Hebrew received three documented wording refinements.

The selected English candidate starts a three-epoch multilingual continuation
at 2e-6, with every epoch exported. All three epochs completed; epoch 3 is
undergoing partner conversation and vision checks. A direct fit comparison on the 48 twelve-turn English demonstrations
found answer-token loss decreased from approximately 4.55 to 3.12 after the
preceding English continuation. Its separate basement/airport conversation still
reused prior answers: 52.8% of turn-5 four-grams appeared in earlier assistant
answers. `chat_probe.py` now records that cross-answer fraction. User turns do
not enter this diagnostic. Sixteen unit tests pass.

The long multilingual stage completed 1,458 example presentations and 93
optimizer updates in 375.35 seconds including exports. Peak allocation was
11.212 GiB. Each saved epoch terminated all fourteen fixed baseline probes.
Russian and Hebrew still contain grammatical defects; validation loss increased
from 3.55687 to 3.99977 across the run. Literal generations, rather than that
loss alone, guide further character selection.

For the same new button/swimming-pool question and seed, epoch 1 at baseline
settings repeated until the token limit; epoch 3 stopped normally. The epoch-3
official text preset (temperature 1.0, top-p 1.0, top-k 20, presence penalty 2.0)
gave a shorter alternative and was retained for a new partner conversation. Its
first scene change, from the pool/wedding to potato/radio, stayed with the new
scene. This differs from the earlier basement/airport test, which used baseline
sampling and another conversation.

A separate cached/uncached comparison used identical history, weights, seed and
baseline settings. Sampled token IDs were identical. Eight greedy cached
next-token choices also matched full-context recomputation; their largest
BF16 logit difference was 0.46875. `check_decode_parity.py` stores the per-step
comparison. `chat_probe.py --no-cache` exposes the recomputation path.

The epoch-3 image check reused the same three photographs and official vision
preset. All twelve outputs are preserved. English descriptions retained the
astronaut suit, tabby cat and espresso cup, while adding detail errors such as
coffee being poured with a spoon. The spoon/saucer and helmet also entered the
character responses. Russian/Hebrew grammar remains weak.

The new Russian partner probe changed from potato/radio to a knotted staircase
at turn 3 without returning to radio; its case and phrasing errors remained.
A fresh Hebrew knot-museum response contained malformed wording and foreign-script
intrusions. These language defects remain part of the next polishing work.
A temporary epoch-3 private Hugging Face backup was completed while partner
evaluation continued.

The epoch-3 English conversation on the official text preset reached twelve
partner turns with the complete history retained. It moved from button/pool
to potato/radio (turn 4), knot museum (turn 7) and cinema/subtitle strike
(turn 11). Turn 10 explicitly returned to the original room-refusing button.
At turn 8, the nameless neck's biography became: “A man who never arrived,
whose arrival was the point.” The partner turns followed the model's actual
answers. They were written by Astra in this session.

The selected continuation candidate remains backed up privately under
`working/long-dialogue-multilingual-epoch003`. Its tested English generation
profile is preserved with the private experiment archive. Final unquantized
and quantized release selection and NoTorch deployment follow the remaining
language work and creator review.

All twelve English answers terminated normally. The largest prompt was 1,079
tokens. Neither within-answer nor prior-assistant-answer four-gram diagnostics
found repeated chains in this particular conversation. The final cinema answer
left “a single frame that never moved” and Dubrovsky becoming “the story itself.”
Raw English, Russian and Hebrew outputs are collected in the private readable
conversation transcript, alongside full JSON histories and generation settings.
